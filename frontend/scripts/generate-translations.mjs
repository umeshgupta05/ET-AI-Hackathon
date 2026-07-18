import { readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { base } from "../src/i18n.js";


const root = resolve(import.meta.dirname, "../..");
const outputPath = resolve(root, "frontend/src/locales/generated.json");
const tempPath = `${outputPath}.tmp`;
const localeNames = {
  mr: "Marathi",
  gu: "Gujarati",
  ml: "Malayalam",
  pa: "Punjabi in Gurmukhi script",
  or: "Odia",
  ur: "Urdu",
};


function flatten(value, prefix = "") {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return child && typeof child === "object" ? flatten(child, path) : [[path, child]];
  });
}


async function loadBackendEnv() {
  const envText = await readFile(resolve(root, "backend/.env"), "utf8").catch(() => "");
  for (const rawLine of envText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const separator = line.indexOf("=");
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim().replace(/^['"]|['"]$/g, "");
    if (!process.env[key]) process.env[key] = value;
  }
}


function validateCatalog(locale, catalog) {
  const required = new Map(flatten(base));
  const translated = new Map(flatten(catalog));
  const missing = [...required.keys()].filter((key) => !translated.has(key));
  const extras = [...translated.keys()].filter((key) => !required.has(key));
  if (missing.length || extras.length) {
    throw new Error(`${locale}: missing [${missing.join(", ")}], extra [${extras.join(", ")}]`);
  }
  const unchanged = [...required].filter(
    ([key, value]) => translated.get(key) === value && !["secondsShort"].includes(key),
  );
  if (unchanged.length > required.size * 0.25) {
    throw new Error(`${locale}: too many untranslated values (${unchanged.length}/${required.size})`);
  }
}


async function translateChunk(locale, language, source, chunkNumber) {
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "openai/gpt-oss-20b",
        temperature: 0.1,
        max_tokens: 2500,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "system",
            content:
              `You are a professional public-safety UI translator. Translate every string value in the supplied JSON into ${language}. ` +
              "Return only valid JSON with exactly the same nested keys and no wrapper object. Keep product/model names, file extensions, API terms, and Python commands unchanged. " +
              "Use concise, respectful language suitable for Indian citizens. Translate fraud, risk, buttons, accessibility labels, errors, and status text naturally.",
          },
          { role: "user", content: JSON.stringify(source) },
        ],
      }),
    });
    if (response.status === 429 && attempt < 3) {
      const waitSeconds = Number(response.headers.get("retry-after")) || 30;
      await new Promise((resolveDelay) => setTimeout(resolveDelay, waitSeconds * 1000));
      continue;
    }
    if (!response.ok) {
      throw new Error(`${locale} chunk ${chunkNumber}: Groq ${response.status} ${await response.text()}`);
    }
    const payload = await response.json();
    const content = payload.choices?.[0]?.message?.content;
    if (!content) throw new Error(`${locale} chunk ${chunkNumber}: model returned no catalog`);
    return JSON.parse(content.replace(/^```json\s*|\s*```$/g, ""));
  }
  throw new Error(`${locale} chunk ${chunkNumber}: retry budget exhausted`);
}


async function translate(locale, language) {
  const entries = Object.entries(base);
  const chunkCount = 6;
  const chunkSize = Math.ceil(entries.length / chunkCount);
  const chunks = Array.from({ length: chunkCount }, (_, index) =>
    Object.fromEntries(entries.slice(index * chunkSize, (index + 1) * chunkSize)),
  ).filter((chunk) => Object.keys(chunk).length);
  const translatedChunks = [];
  for (const [index, chunk] of chunks.entries()) {
    translatedChunks.push(await translateChunk(locale, language, chunk, index + 1));
  }
  const catalog = Object.assign({}, ...translatedChunks);
  const requiredKeys = new Set(flatten(base).map(([key]) => key));
  const translatedKeys = new Set(flatten(catalog).map(([key]) => key));
  const missingRoots = [
    ...new Set(
      [...requiredKeys]
        .filter((key) => !translatedKeys.has(key))
        .map((key) => key.split(".")[0]),
    ),
  ];
  for (const rootKey of missingRoots) {
    const repaired = await translateChunk(
      locale,
      language,
      { [rootKey]: base[rootKey] },
      `repair-${rootKey}`,
    );
    catalog[rootKey] = repaired[rootKey];
  }
  validateCatalog(locale, catalog);
  return catalog;
}


await loadBackendEnv();
if (!process.env.GROQ_API_KEY) {
  throw new Error("GROQ_API_KEY is required to generate translation catalogs");
}

const requested = process.argv.find((arg) => arg.startsWith("--locale="))?.split("=")[1];
const targets = requested ? { [requested]: localeNames[requested] } : localeNames;
if (Object.values(targets).some((name) => !name)) throw new Error(`Unsupported locale: ${requested}`);

const existing = JSON.parse(await readFile(outputPath, "utf8"));
const pending = Object.entries(targets).filter(([locale]) => !existing[locale]);
const results = await Promise.allSettled(
  pending.map(async ([locale, language]) => {
    console.log(`Generating ${language}...`);
    const catalog = await translate(locale, language);
    console.log(`${language}: complete`);
    return [locale, catalog];
  }),
);
for (const result of results) {
  if (result.status === "fulfilled") existing[result.value[0]] = result.value[1];
}
await writeFile(tempPath, `${JSON.stringify(existing, null, 2)}\n`, "utf8");
await rename(tempPath, outputPath);
console.log(`Wrote ${outputPath}`);
const failures = results.filter((result) => result.status === "rejected");
if (failures.length) {
  throw new AggregateError(failures.map((result) => result.reason), "Some locales failed");
}
