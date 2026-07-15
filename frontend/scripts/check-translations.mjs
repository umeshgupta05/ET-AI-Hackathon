import generatedTranslations from "../src/locales/generated.json" with { type: "json" };
import { base, languages, partials } from "../src/i18n.js";


function flatten(value, prefix = "") {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return child && typeof child === "object" ? flatten(child, path) : [[path, child]];
  });
}


const required = new Map(flatten(base));
for (const { code, name } of languages) {
  if (code === "en") continue;
  const overlay = { ...(partials[code] || {}), ...(generatedTranslations[code] || {}) };
  const translated = new Map(flatten(overlay));
  const missing = [...required.keys()].filter((key) => !translated.has(key));
  if (missing.length) throw new Error(`${code} is missing: ${missing.join(", ")}`);
  const unchanged = [...required].filter(
    ([key, value]) => translated.get(key) === value && !["secondsShort"].includes(key),
  );
  if (unchanged.length > required.size * 0.25) {
    throw new Error(`${code} has too many English fallbacks: ${unchanged.length}/${required.size}`);
  }
  console.log(`${name} (${code}): ${required.size}/${required.size} strings covered`);
}

console.log(`Translation coverage: PASS (${languages.length} languages)`);
