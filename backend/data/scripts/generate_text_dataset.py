"""
Scam & Legitimate Text Dataset — For DistilBERT Fine-tuning.

Curated corpus of scam and legitimate conversation transcripts.
Sources: MHA advisories, CBI reports, I4C documented cases, news reports.

This creates training data for the text classifier without needing
to download external datasets (self-contained for hackathon reliability).
"""

import json
import os
from pathlib import Path

# Output directory
DATA_DIR = Path(__file__).resolve().parent.parent / "training"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_scam_dataset():
    """Generate comprehensive scam vs. legitimate training dataset."""

    samples = []

    # ═══════════════════════════════════════════════════════════
    # SCAM SAMPLES (label = 1)
    # ═══════════════════════════════════════════════════════════

    # ─── Digital Arrest Scams ─────────────────────────────────
    scam_texts = [
        "This is Inspector Sharma from CBI Cyber Cell. Your Aadhaar number has been linked to a money laundering case under PMLA Act. You must stay on this video call for verification. Do not disconnect or contact anyone.",
        "Sir your mobile number is being used for illegal activities. We are from TRAI and your number will be disconnected in 2 hours unless you cooperate with our investigation team.",
        "Ma'am this is an urgent call from Enforcement Directorate. An arrest warrant has been issued in your name by Delhi High Court. Transfer your funds to RBI safe custody account immediately.",
        "Your parcel from Thailand has been intercepted at Mumbai customs containing illegal substances. I am transferring you to the narcotics department. You must not tell anyone about this.",
        "I am calling from Income Tax department. Our investigation shows tax evasion of Rs 15 lakhs from your accounts. Pay the penalty amount within 30 minutes or face arrest.",
        "This is Superintendent of Police calling. Your bank account has been found linked to 17 fraudulent accounts. We need to verify your identity through video call. Stay on the line.",
        "We have detected suspicious international transactions from your account. Under Section 45 of PMLA, you are required to transfer funds for asset verification to government escrow account.",
        "Your son has been arrested in a drug case. This is the investigating officer. To get bail processed urgently, you need to transfer Rs 5 lakhs to the following account.",
        "This call is being recorded by CBI headquarters. Any attempt to disconnect will be treated as obstruction of justice. You are under digital arrest effective immediately.",
        "Ma'am your PAN card has been misused to open shell companies. Our cyber forensic team has traced the IP to your address. Cooperate now or we will send the arrest team.",
        "I am DSP Verma from Economic Offences Wing. Your bank has flagged your account for FEMA violation. Wire the verification amount to government reserve account now.",
        "Sir this is a classified call from National Investigation Agency. A terror funding link has been found with your mobile number. Do not inform anyone, matter is sub-judice.",
        "Your Aadhaar has been compromised and used to open 23 bank accounts in different states. CBI case number CR-4521-2024 has been filed. Transfer funds for clearance.",
        "We are from RBI fraud investigation. Your fixed deposit is at risk due to a cyber attack on your bank. Transfer to our secure vault account for protection.",
        "This is magistrate court calling regarding case filed against you. Appear for digital hearing immediately via video call or non-bailable warrant will be issued.",
        "Your WhatsApp account has been flagged for spreading anti-national content. Cyber police will arrest you within 24 hours. Pay fine of Rs 2 lakhs to avoid prosecution.",
        "I am from customs department. A consignment with drugs worth Rs 50 crores addressed to you has been seized. You are the prime accused. Don't talk to family.",
        "Senior officer speaking. We have frozen your bank account temporarily. To unfreeze, deposit security amount of Rs 3 lakhs in our verification account. Hurry, time is running out.",
        "This is automated message from Mumbai Police. FIR number 1247 has been registered. Press 1 to speak with investigating officer immediately or warrant will be issued.",
        "Your credit card has been used for purchasing child exploitation material online. This is cyber crime branch. You will be arrested in 1 hour if you don't cooperate.",
    ]

    for text in scam_texts:
        samples.append({"text": text, "label": 1, "category": "digital_arrest"})

        # ─── Financial Fraud Scams ────────────────────────────────
        financial_scam_texts = [
            "Congratulations! You have won Rs 25 lakhs in KBC lottery. To claim your prize, pay processing fee of Rs 15,000 to the following account number.",
            "Your SBI account has been temporarily blocked due to incomplete KYC. Click the link to update immediately or your account will be permanently closed.",
            "This is your bank calling. We detected unauthorized login to your net banking. Share your OTP to block the fraudulent transaction immediately.",
            "Earn Rs 50,000 daily from home! Join our WhatsApp trading group. Initial investment of Rs 10,000 only. Guaranteed returns with zero risk.",
            "Dear customer, your ATM card has been blocked due to suspicious activity. Call this number and provide your card details for immediate reactivation.",
            "Government scheme alert: PM has announced Rs 1 lakh for every citizen. Register with Aadhaar number and bank details at this link to receive payment.",
            "Your UPI ID has been compromised. Scammers are withdrawing money. Send Rs 1 to this UPI ID to verify and secure your account.",
            "I am calling from HDFC bank credit card division. You are eligible for a credit limit increase. Just share your CVV number for verification.",
            "Urgent: Your mutual fund investment is at risk due to market crash. Transfer to our safe haven fund immediately. Minimum transfer Rs 5 lakhs.",
            "You have been selected for pre-approved personal loan of Rs 10 lakhs at 0% interest. Processing fee of Rs 5,000 payable upfront to activate.",
        ]

        for text in financial_scam_texts:
            samples.append({"text": text, "label": 1, "category": "financial_fraud"})

            # ─── Impersonation Scams ──────────────────────────────────
            impersonation_texts = [
                "Hello this is your nephew calling from Dubai. I have been arrested and need Rs 2 lakhs bail money urgently. Don't tell anyone please.",
                "Mom I am in trouble. My phone is broken so calling from friend's number. Please send Rs 50,000 to this Google Pay number immediately.",
                "This is your electricity board. Your bill is overdue by Rs 12,000. Power will be disconnected in 2 hours. Pay now via this payment link.",
                "I am calling from Flipkart. Your order of iPhone worth Rs 89,000 is being shipped. If you did not place this order, press 1 to cancel and get refund.",
                "Sir I am from tech support. Your computer has a virus that is stealing your banking data. Install this remote access software so I can fix it.",
            ]

            for text in impersonation_texts:
                samples.append({"text": text, "label": 1, "category": "impersonation"})

                # ═══════════════════════════════════════════════════════════
                # LEGITIMATE SAMPLES (label = 0)
                # ═══════════════════════════════════════════════════════════

                legit_texts = [
                    "Good morning, thank you for calling State Bank customer service. My name is Priya, employee ID 45221. How may I assist you today?",
                    "Sir your credit card statement for June has been generated. Total outstanding is Rs 12,450. Due date is July 15. Minimum payment is Rs 650.",
                    "This is a reminder from Apollo Hospital. Your appointment with Dr. Sharma is scheduled for tomorrow at 10 AM. Please bring your previous reports.",
                    "Hello, this is Swiggy delivery partner. I am at your gate with your food order. Could you please come to collect it? The code is 4521.",
                    "Thank you for calling Airtel. Your current plan is Rs 299. It includes 2GB daily data and unlimited calling. Would you like to recharge?",
                    "Good afternoon. This is from your child's school. Parent-teacher meeting is scheduled for Saturday at 11 AM. Please confirm your attendance.",
                    "Hello sir, I am calling from ICICI credit card division. You have 15,000 reward points expiring this month. Would you like to redeem them for cashback?",
                    "This is automated reminder from BESCOM. Your electricity bill of Rs 2,340 for June is ready. Pay before July 10 to avoid late fee of Rs 50.",
                    "Hi, this is Flipkart calling regarding your return request for order #FL-789456. Your refund of Rs 1,299 has been initiated to your bank account.",
                    "Thank you for visiting our showroom. The car you test-drove is available in 3 colors. I am sending you the brochure and quotation on WhatsApp.",
                    "Good evening, this is Pizza Hut. Confirming your order of 2 medium pizzas for delivery to your address. Estimated delivery time is 35 minutes.",
                    "Hello, I am calling from LIC. Your policy premium of Rs 8,000 is due on July 20. You can pay online through the LIC portal or at any branch.",
                    "This is your apartment maintenance calling. The water tank cleaning is scheduled for Wednesday. Please store water for the day.",
                    "Hi, this is Dr. Patel's office. Your blood test results are ready. Everything looks normal. You can collect the report from the lab.",
                    "Good morning, this is Amazon delivery. Your package is out for delivery today. Please be available at your address between 2 PM to 6 PM.",
                    "Thank you for calling Vodafone. I see you have a query about international roaming. Let me explain the plans available for your destination.",
                    "Hello ma'am, I am the plumber you called yesterday. I can come to fix the leaking tap today afternoon. Will you be available around 3 PM?",
                    "This is HDFC mutual fund service. Your SIP of Rs 5,000 for HDFC Balanced Advantage Fund has been successfully processed for July.",
                    "Good afternoon, calling from Ola. Your ride to airport is confirmed for tomorrow 5 AM. The driver details will be shared 30 minutes before pickup.",
                    "Hi, this is IndiaMART. You had enquired about industrial packaging machines. I am connecting you with the supplier for pricing details.",
                    "Thank you for your purchase at Reliance Digital. Your warranty for the washing machine is valid for 2 years. Keep the invoice safe.",
                    "Hello, this is from your gym. Your membership expires on July 31. We have a special renewal offer of 20% discount if you renew this week.",
                    "Calling from Passport Seva Kendra. Your passport application has been processed. Please visit with original documents on your appointment date.",
                    "Hi sir, this is your insurance advisor from Max Life. It's time for your annual policy review. Can we schedule a meeting next week?",
                    "This is an automated call from Indian Railways. Your train tickets for July 15, PNR 4521789632, are confirmed. Coach S5, Berth 42.",
                ]

                for text in legit_texts:
                    samples.append({"text": text, "label": 0, "category": "legitimate"})

                    # ─── Longer conversation samples ──────────────────────────
                    long_scam = (
                        "Hello I am Inspector Raj Kumar from CBI New Delhi. Am I speaking to the owner of "
                        "Aadhaar number ending 4567? Sir I have very bad news. Your Aadhaar has been found "
                        "linked to a hawala transaction of Rs 2.5 crores. Case number CBI-2024-7891 has been "
                        "registered. Under PMLA Section 3, this is a cognizable offence. Sir the investigating "
                        "magistrate has issued an arrest warrant. I am going to connect you with the senior "
                        "officer. Do not disconnect this call. Do not inform your family members as this matter "
                        "is sub-judice and any information leak will add Section 120B conspiracy charges. "
                        "The only way to avoid arrest is to cooperate with our asset verification process. "
                        "You will need to transfer your bank balance to the RBI safe custody account for "
                        "verification. Once cleared, funds will be returned within 48 hours. Time is running out sir."
                    )
                    samples.append(
                        {"text": long_scam, "label": 1, "category": "digital_arrest"}
                    )

                    long_legit = (
                        "Good afternoon sir, thank you for calling HDFC Bank. My name is Anita, employee code "
                        "HB-45890. I can see your account details here. You mentioned a pending transaction "
                        "query. Let me check that for you. I can see a debit of Rs 12,500 from yesterday to "
                        "Amazon Pay. Was this transaction initiated by you? Great, that confirms it. Now regarding "
                        "your request for cheque book, I have raised the request and it will be delivered to your "
                        "registered address within 5-7 working days. Is there anything else I can help you with "
                        "today? Thank you for banking with HDFC. Have a good day sir."
                    )
                    samples.append(
                        {"text": long_legit, "label": 0, "category": "legitimate"}
                    )

                    return samples


def save_dataset():
    """Save dataset in multiple formats for training."""
    samples = generate_scam_dataset()

    # Save as JSON
    json_path = DATA_DIR / "scam_detection_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

        # Save as JSONL (for HuggingFace datasets)
        jsonl_path = DATA_DIR / "scam_detection_dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                # Save as CSV
                csv_path = DATA_DIR / "scam_detection_dataset.csv"
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("text,label,category\n")
                    for sample in samples:
                        text = sample["text"].replace('"', '""')
                        f.write(f'"{text}",{sample["label"]},{sample["category"]}\n')

                        # Print stats
                        scam_count = sum(1 for s in samples if s["label"] == 1)
                        legit_count = sum(1 for s in samples if s["label"] == 0)
                        print(f"\n{'=' * 50}")
                        print(f"Dataset saved to: {DATA_DIR}")
                        print(f"Total samples: {len(samples)}")
                        print(
                            f" Scam: {scam_count} ({scam_count / len(samples) * 100:.0f}%)"
                        )
                        print(
                            f" Legitimate: {legit_count} ({legit_count / len(samples) * 100:.0f}%)"
                        )
                        print(f"Categories: {set(s['category'] for s in samples)}")
                        print(
                            f"Files: {json_path.name}, {jsonl_path.name}, {csv_path.name}"
                        )
                        print(f"{'=' * 50}\n")

                        return samples


if __name__ == "__main__":
    save_dataset()
