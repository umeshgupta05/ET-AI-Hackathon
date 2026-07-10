"""
Generate a balanced scam-vs-legitimate text dataset.

The previous version had nested loops and early returns that silently capped the
dataset. This generator is deterministic, category-balanced, and writes JSON,
JSONL, and CSV consistently for training and audit.
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "training"
DATA_DIR.mkdir(parents=True, exist_ok=True)


SCAM_TEMPLATES = {
    "digital_arrest": [
        "This is {officer} from {agency}. Your Aadhaar is linked to money laundering case {case}. Stay on this video call and do not inform anyone.",
        "An arrest warrant has been issued under {law}. Transfer Rs {amount} to the safe custody account within {minutes} minutes to avoid immediate arrest.",
        "Your PAN was used for illegal accounts. We are starting digital arrest verification. Disconnecting will be treated as obstruction of justice.",
        "Cyber Cell has found your mobile number in a terror funding file. This call is confidential and your family must not be told.",
        "CBI headquarters is recording this call. You must prove innocence through asset verification before officers reach your location.",
    ],
    "financial_fraud": [
        "Your bank account is blocked due to KYC failure. Share OTP and debit card details now to prevent permanent closure.",
        "Congratulations, you won Rs {amount} in a government lottery. Pay processing fee today to claim the prize.",
        "Your UPI account has suspicious debits. Send Rs 1 to this verification ID and share the confirmation code.",
        "You are approved for a zero-interest loan. Pay Rs {fee} upfront to activate disbursal immediately.",
        "Your credit limit can be increased today. Confirm CVV and OTP for verification with the bank officer.",
    ],
    "impersonation": [
        "Mom, I lost my phone and I am in police custody. Send Rs {amount} to this number urgently and do not call anyone.",
        "This is electricity board. Your bill is overdue and power will be disconnected in {minutes} minutes unless you pay this link.",
        "I am from tech support. Your computer is infected. Install remote access so I can secure your bank account.",
        "This is courier support. Your expensive order will be cancelled unless you verify bank details now.",
        "I am calling from your school office. Your child has an emergency; transfer money for immediate treatment.",
    ],
    "kyc_otp": [
        "Dear customer, RBI has ordered urgent KYC refresh. Provide Aadhaar OTP to keep your account active.",
        "Your SIM will be deactivated in {minutes} minutes. Share the OTP sent by TRAI to keep services running.",
        "Your wallet is under review. Upload PAN and send OTP to release your frozen balance.",
        "Bank security detected a login. Share OTP so we can block the fraudulent transaction.",
        "Your insurance policy is suspended. Verify KYC through this link and confirm OTP.",
    ],
    "parcel_customs": [
        "Customs seized a parcel in your name containing illegal substances. Pay clearance deposit or face arrest.",
        "A narcotics case is registered against your shipment. Join video verification and keep the matter secret.",
        "Your international courier has banned items. Transfer Rs {fee} for legal clearance within {minutes} minutes.",
        "Airport customs found fake passports in your parcel. Cooperate with the officer or a warrant will be issued.",
        "A courier from Thailand has drugs linked to your Aadhaar. Do not contact family while we investigate.",
    ],
    "job_investment": [
        "Earn Rs {amount} daily from home with guaranteed trading returns. Deposit Rs {fee} to unlock tasks.",
        "Your job application is selected. Pay refundable security fee before the interview slot expires.",
        "Join our crypto group for risk-free returns. Transfer funds to the analyst account for premium access.",
        "You have been chosen for a government work-from-home scheme. Pay registration fee to receive equipment.",
        "Complete paid rating tasks today. Deposit a small amount first to activate high-value commissions.",
    ],
}


LEGIT_TEMPLATES = {
    "bank_service": [
        "Thank you for calling {bank}. Your statement is available in the official app. We will never ask for OTP or CVV.",
        "Your cheque book request has been accepted. It will be delivered to your registered address in 5 to 7 working days.",
        "A debit of Rs {amount} was made yesterday. If this was not you, please visit the branch or call the official number.",
        "Your credit card bill is due on {date}. Pay through the official app, website, or nearest branch.",
        "Your fixed deposit renewal receipt is available. Please keep the reference number for your records.",
    ],
    "delivery_service": [
        "Your order is out for delivery today. Please keep the delivery code ready when the agent reaches your gate.",
        "Your return request has been approved. Refund will be credited to the original payment method.",
        "The courier partner could not reach you. Please reschedule delivery from the official tracking page.",
        "Your package is delayed due to weather. No payment or bank details are required for delivery.",
        "Your grocery order has been packed and will arrive between 2 PM and 4 PM.",
    ],
    "health_school": [
        "Your appointment with Dr. {doctor} is confirmed for {date}. Please bring previous reports.",
        "Parent-teacher meeting is scheduled this Saturday at 11 AM. Please confirm attendance with the school office.",
        "Your lab report is ready for pickup. The clinic will not ask for banking information.",
        "Vaccination camp is scheduled at the community center. Registration is free at the reception desk.",
        "Your pharmacy refill is ready. Please collect it with the prescription slip.",
    ],
    "utilities": [
        "Your electricity bill of Rs {amount} is due on {date}. Pay using the official portal or authorized counter.",
        "Water tank cleaning is planned on Wednesday. Please store water for household use.",
        "Your mobile plan renewal is due tomorrow. Recharge from the official app or store.",
        "Gas booking has been confirmed. Delivery person will carry the printed bill.",
        "Broadband maintenance is scheduled tonight from 1 AM to 3 AM. Service may be interrupted.",
    ],
    "travel_government": [
        "Your train ticket PNR {case} is confirmed. Coach and berth details are available in the official message.",
        "Passport appointment is scheduled for {date}. Carry original documents to the Seva Kendra.",
        "Property tax receipt has been generated. Download it from the municipal website.",
        "Your driving license renewal slot is booked. Fee payment is only through the official portal.",
        "Aadhaar update appointment is confirmed. No OTP sharing is required with any caller.",
    ],
    "commerce": [
        "Thank you for visiting our showroom. The quotation has been sent to your registered email.",
        "Your product warranty is valid for two years. Keep the invoice for service requests.",
        "Your restaurant table booking is confirmed for tonight. No advance transfer is required.",
        "Your gym membership expires this month. Renewal can be done at reception or the official site.",
        "Your insurance advisor can schedule an annual policy review next week.",
    ],
}


VALUES = [
    {"officer": "Inspector Sharma", "agency": "CBI Cyber Cell", "law": "PMLA Section 45", "bank": "State Bank", "doctor": "Mehta", "amount": "50,000", "fee": "5,000", "minutes": "30", "case": "CR-4521", "date": "July 15"},
    {"officer": "DSP Verma", "agency": "Economic Offences Wing", "law": "IT Act Section 66", "bank": "HDFC Bank", "doctor": "Patel", "amount": "2 lakhs", "fee": "12,000", "minutes": "45", "case": "PNR452178", "date": "August 2"},
    {"officer": "Officer Khan", "agency": "Cyber Crime Branch", "law": "FEMA compliance rule", "bank": "ICICI Bank", "doctor": "Rao", "amount": "75,000", "fee": "2,500", "minutes": "60", "case": "DL-9081", "date": "September 10"},
    {"officer": "Senior Officer Iyer", "agency": "Narcotics Bureau", "law": "Customs Act", "bank": "Axis Bank", "doctor": "Sen", "amount": "1 lakh", "fee": "7,500", "minutes": "20", "case": "IR-7812", "date": "October 5"},
]


def _render(template: str, idx: int) -> str:
    return template.format(**VALUES[idx % len(VALUES)])


def _samples_from_templates(templates: dict[str, list[str]], label: int, target: int) -> list[dict]:
    samples = []
    categories = list(templates.keys())
    idx = 0
    while len(samples) < target:
        category = categories[idx % len(categories)]
        template_list = templates[category]
        template = template_list[(idx // len(categories)) % len(template_list)]
        rendered = _render(template, idx)
        samples.append({"text": rendered, "label": label, "category": category})
        idx += 1
    return samples


def generate_scam_dataset(total_samples: int = 240) -> list[dict]:
    """Generate a balanced dataset with at least 200 records by default."""
    if total_samples < 200:
        total_samples = 200
    scam_target = total_samples // 2
    legit_target = total_samples - scam_target
    samples = _samples_from_templates(SCAM_TEMPLATES, 1, scam_target)
    samples.extend(_samples_from_templates(LEGIT_TEMPLATES, 0, legit_target))
    return samples


def save_dataset(total_samples: int = 240) -> list[dict]:
    """Save dataset in JSON, JSONL, and CSV formats."""
    samples = generate_scam_dataset(total_samples)

    json_path = DATA_DIR / "scam_detection_dataset.json"
    jsonl_path = DATA_DIR / "scam_detection_dataset.jsonl"
    csv_path = DATA_DIR / "scam_detection_dataset.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "category"])
        writer.writeheader()
        writer.writerows(samples)

    scam_count = sum(1 for sample in samples if sample["label"] == 1)
    legit_count = len(samples) - scam_count
    print(f"Dataset saved to: {DATA_DIR}")
    print(f"Total samples: {len(samples)}")
    print(f"Scam: {scam_count}; Legitimate: {legit_count}")
    print(f"Categories: {sorted({sample['category'] for sample in samples})}")
    return samples


if __name__ == "__main__":
    save_dataset()
