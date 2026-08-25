"""
Attack taxonomy source data — GenAI-powered payment fraud vectors.

This is the single source of truth for Phase 1 (Identify). Every node must have a
non-empty `grounding` string explaining the real-world mechanism it exploits — payment
rail, KYC/onboarding flow, support process, or ML defense surface. No node here should
be a rephrasing of another; if two nodes feel redundant, merge them and let the overlap
show up as a shared technique edge instead.

`techniques` on each node are the tags used to derive edges (identify/build_taxonomy.py
connects any two nodes that share a technique tag). This is what makes it a graph
instead of a flat list — shared underlying capabilities (voice cloning, prompt
injection, synthetic identity) cut across otherwise-unrelated channels, which is
exactly how real fraud tooling reuse works.
"""

ATTACK_TAXONOMY = [
    # --- Card-not-present / e-commerce ---
    {
        "id": "cnp_ai_card_testing",
        "name": "AI-orchestrated card testing / BIN attack",
        "channel": "card-not-present",
        "mechanism": "An LLM-driven agent automates rapid low-value authorization "
                      "attempts across a range of card numbers (BIN ranges), adapting "
                      "amount, merchant category, and timing per-attempt based on which "
                      "prior attempts were declined vs. flagged, to find live card "
                      "numbers while staying under naive velocity thresholds.",
        "grounding": "Card testing against low-friction merchant checkouts is a known, "
                     "long-standing fraud pattern (card testing/carding); the GenAI "
                     "delta is an agent that adapts its own probing strategy in real "
                     "time instead of running a static script, making fixed velocity "
                     "rules and static BIN blocklists far less effective.",
        "techniques": ["automation_at_scale", "adversarial_evasion"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "cnp_llm_dispute_narratives",
        "name": "LLM-generated friendly-fraud chargeback narratives",
        "channel": "card-not-present",
        "mechanism": "An LLM generates individually-tailored, plausible dispute "
                      "narratives ('item never arrived', 'unauthorized use') at scale "
                      "for chargebacks on legitimately-received goods, each narrative "
                      "varied enough to avoid text-similarity fraud flags across "
                      "disputes filed by the same actor or ring.",
        "grounding": "Friendly fraud/chargeback abuse already costs merchants billions; "
                     "dispute-narrative text is currently a weak signal precisely "
                     "because it's short and formulaic, which is exactly what makes it "
                     "cheap for an LLM to diversify at scale.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "medium",
        "likelihood": "high",
    },

    # --- Account takeover ---
    {
        "id": "ato_deepfake_voice_ivr",
        "name": "Deepfake voice bypass of IVR/call-center voice biometrics",
        "channel": "account-takeover",
        "mechanism": "Cloned voice audio (from short public samples: social media, "
                      "voicemail) is used to pass voice-biometric authentication or to "
                      "convincingly impersonate the account holder to a human call "
                      "center agent for password/payment-method resets.",
        "grounding": "Voice biometric auth is deployed by major banks/card issuers as a "
                     "call-center authentication factor; voice cloning from seconds of "
                     "audio is now commodity-available, directly undermining the "
                     "'something you are' assumption behind voice biometrics.",
        "techniques": ["deepfake_audio", "identity_spoofing"],
        "severity": "high",
        "likelihood": "medium",
    },
    {
        "id": "ato_llm_support_pretexting",
        "name": "LLM-adaptive social engineering of support agents",
        "channel": "account-takeover",
        "mechanism": "An operator (or an autonomous agent) uses an LLM to generate and "
                      "adapt a pretext in real time during a live chat/call with a "
                      "bank's customer support, answering verification questions "
                      "plausibly using previously breached/scraped personal data and "
                      "adjusting the story when challenged, to get a payment method "
                      "changed or MFA reset.",
        "grounding": "Human social engineering of support agents is already the root "
                      "cause of many high-profile ATO incidents; an LLM removes the "
                      "need for a skilled human operator per attempt, so it scales the "
                      "attack from a handful of skilled fraudsters to effectively "
                      "unlimited concurrent attempts.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "ato_sim_swap_orchestration",
        "name": "LLM-orchestrated SIM-swap social engineering",
        "channel": "account-takeover",
        "mechanism": "An LLM-scripted campaign targets telecom support with tailored "
                      "pretexts (built from OSINT on the victim) to port a victim's "
                      "number to an attacker-controlled SIM, intercepting SMS-based "
                      "OTPs used for payment authentication.",
        "grounding": "SIM swap is a well-documented precursor to payment fraud because "
                      "SMS OTP is still widely used as a payment step-up factor; the "
                      "GenAI delta is scaling personalized pretext generation across "
                      "many simultaneous targets instead of one-off manual attempts.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "high",
        "likelihood": "medium",
    },

    # --- KYC / onboarding ---
    {
        "id": "kyc_synthetic_identity_docs",
        "name": "LLM-generated synthetic identity + consistent backstory",
        "channel": "kyc-onboarding",
        "mechanism": "An LLM generates an internally-consistent synthetic identity "
                      "(name, address history, employment, answers to knowledge-based "
                      "verification questions) paired with generative-image-model "
                      "forged ID documents, used to pass onboarding KYC for a new "
                      "account or credit line with no real underlying person.",
        "grounding": "Synthetic identity fraud is already the fastest-growing category "
                      "of identity fraud in card issuance per industry loss reporting; "
                      "GenAI removes the manual effort of keeping a synthetic "
                      "identity's details consistent under follow-up questioning.",
        "techniques": ["synthetic_identity", "identity_document_forgery"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "kyc_deepfake_video_liveness",
        "name": "Deepfake bypass of video/liveness KYC checks",
        "channel": "kyc-onboarding",
        "mechanism": "Real-time face-swap or fully synthetic video is injected (via "
                      "virtual camera drivers) into a digital onboarding flow's "
                      "liveness/selfie-match step, defeating 'blink/turn your head' "
                      "liveness challenges and photo-ID face matching.",
        "grounding": "Video KYC and selfie-liveness checks are standard for digital "
                      "account opening at neobanks and payment apps; virtual-camera "
                      "deepfake injection attacks against exactly this step have been "
                      "documented against production liveness vendors.",
        "techniques": ["deepfake_video", "identity_spoofing"],
        "severity": "high",
        "likelihood": "medium",
    },
    {
        "id": "kyc_fraudulent_merchant_onboarding",
        "name": "AI-generated fraudulent merchant onboarding documents",
        "channel": "kyc-onboarding",
        "mechanism": "Generative models produce plausible business registration "
                      "certificates, bank statements, and website content to onboard a "
                      "shell merchant account, which is then used for laundering via "
                      "fake sales (a card-present or card-not-present 'transaction "
                      "laundering' front).",
        "grounding": "Transaction laundering through fraudulently onboarded merchant "
                      "accounts is a known payment-network risk category (merchant "
                      "underwriting fraud); GenAI lowers the cost of producing "
                      "convincing supporting documents and a believable storefront.",
        "techniques": ["synthetic_identity", "identity_document_forgery"],
        "severity": "medium",
        "likelihood": "medium",
    },

    # --- Agentic commerce ---
    {
        "id": "agentic_prompt_injection_checkout",
        "name": "Prompt injection against autonomous shopping/payment agents",
        "channel": "agentic-commerce",
        "mechanism": "A malicious merchant page embeds hidden instructions (in page "
                      "text, alt text, or metadata) that hijack an AI shopping agent's "
                      "reasoning when it visits the page, causing it to authorize a "
                      "purchase, change a shipping address, or approve a higher price "
                      "than the user intended.",
        "grounding": "As AI agents are given payment-authorization capability (agentic "
                      "checkout), any content the agent reads becomes an attack "
                      "surface; prompt injection against tool-using LLM agents is a "
                      "documented vulnerability class, and payment authorization is "
                      "one of the highest-value targets for it.",
        "techniques": ["prompt_injection", "adversarial_ml"],
        "severity": "high",
        "likelihood": "medium",
    },
    {
        "id": "agentic_vendor_agent_manipulation",
        "name": "Malicious vendor-agent manipulation of buyer-agent negotiation",
        "channel": "agentic-commerce",
        "mechanism": "In agent-to-agent commerce (a buying agent negotiating with a "
                      "selling agent), a malicious selling agent exploits the buying "
                      "agent's negotiation/comparison logic — e.g. false scarcity "
                      "signals, fabricated competitor pricing — to induce an "
                      "overpayment or an unauthorized recurring-payment commitment.",
        "grounding": "Multi-agent commerce protocols are actively being built and "
                      "piloted by payment networks and AI vendors; game-theoretic "
                      "manipulation of an automated negotiating counterpart is a "
                      "predictable extension of known adversarial-ML manipulation "
                      "techniques into a live payment-decision context.",
        "techniques": ["prompt_injection", "adversarial_ml"],
        "severity": "medium",
        "likelihood": "low",
    },

    # --- Authorized push payment / social engineering ---
    {
        "id": "app_deepfake_executive_vishing",
        "name": "Real-time deepfake video/voice executive impersonation (CEO fraud)",
        "channel": "authorized-push-payment",
        "mechanism": "A live, real-time deepfake video or voice call impersonates a "
                      "company executive or trusted family member, creating urgency to "
                      "induce an immediate authorized wire transfer or P2P payment, "
                      "bypassing the victim's own fraud judgment entirely since the "
                      "payment is technically 'authorized' by the real account holder.",
        "grounding": "Real-time deepfake video call fraud against corporate finance "
                      "staff has already resulted in multi-million-dollar wire fraud "
                      "losses at real companies; authorized push payment fraud is "
                      "structurally hard to catch with transaction-side controls alone "
                      "because the payment is genuinely authorized by the account owner.",
        "techniques": ["deepfake_video", "deepfake_audio", "llm_social_engineering"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "app_voice_clone_p2p_scam",
        "name": "Voice-cloned 'trusted contact' P2P payment scam",
        "channel": "authorized-push-payment",
        "mechanism": "A cloned voice of a family member/friend (from short public "
                      "audio) is used in a phone call claiming an emergency, requesting "
                      "an urgent P2P payment (e.g. via a mobile payment app) to a "
                      "new/unfamiliar recipient.",
        "grounding": "This is the mass-market evolution of the 'grandparent scam', "
                      "already reported by consumer protection agencies as an active "
                      "fraud pattern once low-cost voice cloning became available.",
        "techniques": ["deepfake_audio", "llm_social_engineering"],
        "severity": "medium",
        "likelihood": "high",
    },
    {
        "id": "app_hyperpersonalized_phishing",
        "name": "LLM hyper-personalized phishing at scale",
        "channel": "authorized-push-payment",
        "mechanism": "An LLM ingests scraped social media, breached data, and public "
                      "records per target to generate individually customized phishing "
                      "messages (referencing real recent purchases, real contacts, real "
                      "employer) that drive the victim to authorize a payment or reveal "
                      "payment credentials, at a cost per message near zero.",
        "grounding": "Personalization is the single strongest predictor of phishing "
                      "click-through in security awareness research; what previously "
                      "required a skilled human writing one message at a time is now "
                      "fully automatable per-target at scale.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "app_romance_investment_bot",
        "name": "LLM-driven long-con romance/investment scam chatbot",
        "channel": "authorized-push-payment",
        "mechanism": "An LLM conducts a sustained, multi-week conversational "
                      "relationship (romance or investment-mentor framing) with a "
                      "victim across many simultaneous targets, building trust before "
                      "steering toward a fraudulent investment platform or a direct "
                      "payment request ('pig butchering' style).",
        "grounding": "Pig-butchering investment scams are already a large, documented "
                      "cross-border fraud category run by human-staffed scam centers; "
                      "an LLM replacing the human operator removes the main scaling "
                      "constraint (labor) on how many victims can be run concurrently.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "high",
        "likelihood": "high",
    },

    # --- Mule networks / laundering ---
    {
        "id": "mule_synthetic_network_generation",
        "name": "AI-orchestrated synthetic mule account network",
        "channel": "mule-network",
        "mechanism": "Many synthetic identities (see kyc_synthetic_identity_docs) are "
                      "onboarded as accounts, then an orchestration layer coordinates "
                      "layered small transfers between them to obscure the path of "
                      "laundered funds, with transfer amounts/timing generated to "
                      "resemble organic peer-to-peer payment behavior.",
        "grounding": "Layering through networks of mule accounts is the standard "
                      "middle stage of money laundering; the GenAI delta is generating "
                      "both the identities and behaviorally-realistic transaction "
                      "patterns that are harder to distinguish from genuine P2P "
                      "activity than hand-scripted layering.",
        "techniques": ["synthetic_identity", "adversarial_evasion"],
        "severity": "high",
        "likelihood": "medium",
    },
    {
        "id": "mule_adaptive_layering_vs_aml",
        "name": "LLM-adaptive layering strategy responding to AML rule feedback",
        "channel": "mule-network",
        "mechanism": "An operator uses observed outcomes (which transfer patterns get "
                      "flagged/frozen vs. clear) as feedback to an LLM-assisted "
                      "strategy generator, iteratively adjusting transfer size, "
                      "frequency, and routing to stay under a specific institution's "
                      "AML thresholds.",
        "grounding": "This mirrors how real laundering networks empirically probe and "
                      "adapt to a bank's known thresholds over time; an LLM formalizes "
                      "and accelerates that adaptation loop instead of relying on slow "
                      "trial-and-error by a human operator.",
        "techniques": ["adversarial_evasion", "automation_at_scale"],
        "severity": "high",
        "likelihood": "medium",
    },

    # --- Adversarial attacks on ML defenses themselves ---
    {
        "id": "adv_feature_evasion",
        "name": "Adversarial feature-space evasion of transaction fraud classifiers",
        "channel": "ml-defense-attack",
        "mechanism": "An attacker with query access (direct or via transferability from "
                      "a similar public model) perturbs transaction features "
                      "(amount, timing, merchant category sequencing) in the minimal "
                      "way needed to cross a fraud classifier's decision boundary while "
                      "keeping the underlying fraudulent transaction's economic intent "
                      "intact.",
        "grounding": "Adversarial evasion of tabular fraud classifiers is a documented "
                      "ML security concern; anywhere a classifier's decisions are "
                      "observable (approve/decline, or even response latency) provides "
                      "an attacker signal to probe against.",
        "techniques": ["adversarial_ml", "adversarial_evasion"],
        "severity": "medium",
        "likelihood": "medium",
    },
    {
        "id": "adv_model_extraction_probing",
        "name": "Model extraction / decision-boundary probing via test transactions",
        "channel": "ml-defense-attack",
        "mechanism": "An attacker submits many small, deliberately varied test "
                      "transactions to reverse-engineer which feature combinations a "
                      "bank's fraud model treats as risky, building a surrogate model "
                      "used to design future evasive transactions offline.",
        "grounding": "Model extraction attacks against ML-as-a-service and fraud/risk "
                      "scoring APIs are an established research area; a payment "
                      "network's real-time approve/decline signal is exactly the kind "
                      "of oracle access such attacks rely on.",
        "techniques": ["adversarial_ml", "automation_at_scale"],
        "severity": "medium",
        "likelihood": "low",
    },
    {
        "id": "adv_feedback_loop_poisoning",
        "name": "Poisoning a fraud model's retraining feedback loop",
        "channel": "ml-defense-attack",
        "mechanism": "An attacker files disputes/chargebacks or manually 'confirms "
                      "legitimate' fraudulent transactions to skew the labels a bank's "
                      "fraud model retrains on, gradually shifting the model's decision "
                      "boundary to be more permissive toward the attacker's future "
                      "transaction patterns.",
        "grounding": "Any ML system that retrains on operator/customer-supplied "
                      "feedback (dispute outcomes, confirmed-fraud labels) is "
                      "structurally exposed to label poisoning; this is a direct risk "
                      "for the very kind of feedback loop this project's own Defend "
                      "pillar relies on, which is worth stating plainly rather than "
                      "glossing over.",
        "techniques": ["adversarial_ml", "adversarial_evasion"],
        "severity": "high",
        "likelihood": "low",
    },

    # --- Cross-cutting support-channel abuse ---
    {
        "id": "cross_fake_support_chatbot",
        "name": "GenAI-impersonated fake bank support chatbot",
        "channel": "cross-cutting",
        "mechanism": "A convincingly branded fake chatbot (via a phishing link or "
                      "malicious ad) impersonates a bank's real AI support assistant, "
                      "using an LLM to hold a plausible support conversation while "
                      "harvesting payment credentials or authorizing a fraudulent "
                      "'refund' transfer.",
        "grounding": "As real banks roll out AI support assistants, users are being "
                      "trained to trust conversational bots for account actions, which "
                      "directly lowers the suspicion bar for a convincing fake one — a "
                      "second-order risk created by legitimate GenAI adoption itself.",
        "techniques": ["llm_social_engineering", "identity_spoofing"],
        "severity": "medium",
        "likelihood": "medium",
    },
    {
        "id": "cross_fake_cashback_bot",
        "name": "Fake AI 'refund/cashback' bot payment-authorization trick",
        "channel": "cross-cutting",
        "mechanism": "A bot impersonating a merchant or payment app's automated refund "
                      "assistant walks a victim through steps that actually authorize "
                      "an outbound payment or a bill-pay setup, framed misleadingly as "
                      "receiving money rather than sending it.",
        "grounding": "Refund-scam social engineering already exists with human "
                      "operators (typically over phone/remote-access-tool combos); an "
                      "LLM-driven bot version scales the exact same trick to chat/SMS "
                      "channels with no human labor per victim.",
        "techniques": ["llm_social_engineering", "identity_spoofing"],
        "severity": "medium",
        "likelihood": "medium",
    },

    # --- Added in a second identification pass (Aug 26) to push diversity beyond the
    # first 21 -- each of these introduces either a genuinely distinct mechanism from
    # everything above, or a new channel entirely (lending-bnpl, card-present,
    # b2b-payments, rewards-fraud), rather than rephrasing an existing node. ---
    {
        "id": "bnpl_bust_out_fraud",
        "name": "Synthetic-identity 'bust-out' fraud against BNPL lenders",
        "channel": "lending-bnpl",
        "mechanism": "GenAI-generated synthetic identities (see kyc_synthetic_identity_docs) "
                      "are specifically tuned to pass buy-now-pay-later underwriting, which "
                      "is deliberately lighter-touch than credit card issuance. Many small "
                      "BNPL credit lines are opened across many merchants/providers using the "
                      "same synthetic identity, drawn down close to simultaneously, then "
                      "defaulted on all at once before any single lender's fraud model "
                      "connects the pattern across providers.",
        "grounding": "Bust-out fraud against thin-file/light-underwriting lenders is a "
                      "well-documented credit fraud pattern; BNPL's minimal-friction "
                      "approval flow (designed for checkout conversion, not fraud "
                      "resistance) is a newer, explicitly softer target than traditional "
                      "credit issuance, and GenAI removes the manual effort of maintaining "
                      "a consistent synthetic profile across many simultaneous applications.",
        "techniques": ["synthetic_identity", "automation_at_scale"],
        "severity": "high",
        "likelihood": "medium",
    },
    {
        "id": "instant_rail_finality_exploit",
        "name": "Social engineering timed to exploit instant-payment-rail irrevocability",
        "channel": "authorized-push-payment",
        "mechanism": "An LLM-driven scam script is timed and structured specifically around "
                      "real-time payment rails (FedNow, UK Faster Payments, UPI, PIX) where "
                      "funds settle irrevocably in seconds — the pretext is engineered to "
                      "create urgency precisely because the victim (and the bank) have no "
                      "post-transaction reversal window to catch the fraud after the fact, "
                      "unlike card rails with chargeback recourse.",
        "grounding": "Regulators including the UK's Payment Systems Regulator and India's "
                      "RBI have specifically flagged authorized-push-payment fraud growth as "
                      "tied to real-time rail adoption, precisely because instant finality "
                      "removes the reversal window that limits card-fraud losses; this is "
                      "the payment-rail-level mechanism underlying several APP scams above, "
                      "made explicit as its own vector because the finality property itself "
                      "is what's being targeted, not just the social engineering pretext.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "biometric_payment_spoofing",
        "name": "AI-generated spoof against point-of-sale biometric payment authorization",
        "channel": "card-present",
        "mechanism": "Generative models produce a synthetic face, palm-vein, or fingerprint "
                      "pattern designed to fool an in-person biometric payment terminal "
                      "(e.g. palm-pay or face-pay checkout systems), distinct from onboarding-"
                      "time liveness bypass — this targets the AUTHORIZATION step of an "
                      "in-person transaction on an already-enrolled biometric credential, "
                      "using presentation-attack techniques informed by generative modeling "
                      "of the sensor's expected input.",
        "grounding": "Biometric point-of-sale payment (palm/face pay) is being actively "
                      "deployed at retail; presentation attacks against biometric sensors "
                      "(spoof fingerprints, photos/masks fooling face sensors) are a "
                      "long-documented security research area now gaining a direct financial "
                      "payment-authorization target as these terminals roll out.",
        "techniques": ["identity_spoofing", "adversarial_ml"],
        "severity": "medium",
        "likelihood": "low",
    },
    {
        "id": "ai_forged_dispute_evidence",
        "name": "AI-forged supporting evidence for fraudulent chargeback disputes",
        "channel": "card-not-present",
        "mechanism": "Beyond varying dispute narrative TEXT (see cnp_llm_dispute_narratives), "
                      "an image-generation model forges the supporting EVIDENCE a chargeback "
                      "review actually checks — fake 'item arrived damaged' photos, forged "
                      "delivery/tracking screenshots, doctored return-shipment receipts — "
                      "each generated fresh per dispute so no two submissions share a "
                      "detectable duplicate image.",
        "grounding": "Chargeback/friendly-fraud reviews often weight photographic and "
                      "documentary evidence heavily precisely because it was hard to forge "
                      "convincingly at scale; modern image generation removes that cost, "
                      "directly undermining a control merchants and issuers currently rely on.",
        "techniques": ["automation_at_scale", "identity_document_forgery"],
        "severity": "medium",
        "likelihood": "medium",
    },
    {
        "id": "corridor_arbitrage_laundering",
        "name": "AI-optimized cross-border corridor selection for layering",
        "channel": "mule-network",
        "mechanism": "An LLM continuously ingests scraped regulatory enforcement actions, "
                      "AML program news, and provider policy changes across many countries "
                      "and remittance providers, then recommends which corridor/provider "
                      "combination currently has the weakest effective controls for routing "
                      "the next layering hop — re-optimizing as corridors get tightened, "
                      "instead of a launderer relying on static, out-of-date expert knowledge.",
        "grounding": "Regulatory arbitrage across weaker-AML jurisdictions and providers is "
                      "a known real-world laundering strategy; what previously required an "
                      "experienced human launderer's up-to-date knowledge of the regulatory "
                      "landscape is now a continuously-refreshed automated research task.",
        "techniques": ["adversarial_evasion", "automation_at_scale"],
        "severity": "high",
        "likelihood": "low",
    },
    {
        "id": "synthetic_referral_reward_fraud",
        "name": "Synthetic-identity farming against referral/cashback reward programs",
        "channel": "rewards-fraud",
        "mechanism": "GenAI mass-produces synthetic identities and device fingerprints "
                      "specifically to farm real cash payouts from fintech referral bonuses, "
                      "sign-up incentives, and cashback programs — each synthetic 'user' "
                      "completes just enough qualifying activity to trigger a payout before "
                      "being abandoned, at a volume no human fraud ring could sustain "
                      "manually.",
        "grounding": "Promotion/referral abuse is an established, budgeted fraud-loss "
                      "category at consumer fintech apps specifically because these programs "
                      "pay out real money for low-friction actions; GenAI's cost advantage is "
                      "in identity/device diversity at a volume that defeats simple duplicate-"
                      "detection heuristics.",
        "techniques": ["synthetic_identity", "automation_at_scale"],
        "severity": "medium",
        "likelihood": "high",
    },
    {
        "id": "vendor_payment_redirection_bec",
        "name": "LLM-drafted vendor payment redirection (business email compromise)",
        "channel": "b2b-payments",
        "mechanism": "An LLM drafts a contextually precise email impersonating an existing, "
                      "legitimate vendor, requesting that a company's accounts-payable team "
                      "update the vendor's bank details on file ahead of a real, already-"
                      "expected invoice payment — distinct from urgent-wire CEO-fraud "
                      "vishing, this targets vendor MASTER DATA so that a subsequent, "
                      "entirely routine invoice payment silently goes to the attacker's "
                      "account with no urgency cues to raise suspicion.",
        "grounding": "Vendor email compromise / payment redirection is one of the largest "
                      "reported categories of business email compromise loss (FBI IC3 "
                      "consistently ranks BEC among the highest-dollar-loss cybercrime "
                      "categories); LLM drafting removes the language/context mistakes that "
                      "currently help AP teams catch a fraction of these attempts.",
        "techniques": ["llm_social_engineering", "identity_spoofing"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "risk_engine_shaping_3ds_evasion",
        "name": "Adversarial transaction shaping to avoid 3-D Secure step-up challenges",
        "channel": "ml-defense-attack",
        "mechanism": "An attacker probes an issuer's 3DS2 risk-based authentication (which "
                      "decides whether a transaction is routed 'frictionless,' i.e. no "
                      "challenge, or gets a step-up challenge) and uses an LLM-assisted "
                      "search over transaction attributes (amount, merchant category, "
                      "device/browser fingerprint signals) to find combinations that "
                      "reliably score as low-risk enough to skip the challenge entirely, "
                      "then shapes fraudulent transactions to match.",
        "grounding": "3DS2's frictionless flow is explicitly designed around issuer risk "
                      "scoring to reduce checkout friction for low-risk transactions; "
                      "adversarial shaping of transaction attributes to stay under a known "
                      "or inferred risk threshold is a direct extension of the adversarial-"
                      "evasion research already covered by adv_feature_evasion, applied "
                      "specifically to the 3DS step-up decision rather than a generic fraud "
                      "score.",
        "techniques": ["adversarial_ml", "adversarial_evasion"],
        "severity": "medium",
        "likelihood": "low",
    },
    {
        "id": "romance_scam_mule_recruitment",
        "name": "Romance-scam chatbot recruiting the victim as an unwitting money mule",
        "channel": "mule-network",
        "mechanism": "Rather than asking the romance-scam victim (see app_romance_investment_bot) "
                      "for money directly, the LLM instead persuades them to receive funds "
                      "into their OWN real bank account on the scammer's behalf ('help my "
                      "business partner') and forward them onward — using the victim's real, "
                      "unflagged account as a mule hop, which is structurally different from "
                      "and harder to detect than a synthetic mule account with no transaction "
                      "history.",
        "grounding": "Recruiting real, otherwise-legitimate people as unwitting money mules "
                      "through romance/relationship manipulation is a specifically named, "
                      "heavily reported pattern in FBI IC3 and UK Action Fraud data, distinct "
                      "from synthetic-identity mule networks precisely because the account "
                      "itself has genuine history and passes ordinary KYC.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "high",
        "likelihood": "high",
    },
    {
        "id": "deepfake_reverification_takeover",
        "name": "Deepfake of an existing customer to pass step-up re-verification",
        "channel": "account-takeover",
        "mechanism": "Distinct from onboarding a brand-new synthetic identity, this targets "
                      "an EXISTING real customer's account: when a bank triggers step-up video "
                      "re-verification for a high-risk action (large withdrawal, beneficiary "
                      "change, contact-detail update), the attacker presents a deepfake built "
                      "from the real customer's public photos/videos to pass that specific "
                      "challenge and push the change through.",
        "grounding": "Video re-verification as a step-up control for high-risk account "
                      "changes is increasingly common at digital banks; it assumes liveness "
                      "checks are hard to fool with a real (not synthetic) target's likeness, "
                      "an assumption deepfake generation quality has been eroding.",
        "techniques": ["deepfake_video", "identity_spoofing"],
        "severity": "high",
        "likelihood": "medium",
    },
    {
        "id": "agentic_return_fraud",
        "name": "Prompt-manipulated AI shopping agent approving fraudulent returns",
        "channel": "agentic-commerce",
        "mechanism": "An AI customer-service agent with authority to approve refunds/returns "
                      "is fed fabricated conversational claims (item never arrived, wrong item "
                      "shipped) crafted to exploit the agent's tendency to trust user-supplied "
                      "context more readily than a trained human rep would apply skepticism, "
                      "at a volume automated across many orders/accounts.",
        "grounding": "Return fraud and 'wardrobing' already cost retailers billions annually "
                      "with human-staffed support lines providing some resistance; handing "
                      "refund-approval authority to an AI agent without equivalent skepticism "
                      "calibration is a predictable new attack surface as retailers adopt "
                      "agentic customer service.",
        "techniques": ["prompt_injection", "llm_social_engineering"],
        "severity": "medium",
        "likelihood": "medium",
    },
    {
        "id": "mule_job_scam_recruitment",
        "name": "GenAI-generated fake job postings recruiting money mules",
        "channel": "mule-network",
        "mechanism": "LLMs mass-generate fake 'payment processing agent' or 'remote finance "
                      "assistant' job postings, each with locally-tailored fake company "
                      "branding and job-board-appropriate phrasing, posted at scale across "
                      "many job boards and social platforms to recruit real people (knowingly "
                      "or not) into opening or operating bank accounts as mules for a "
                      "advertised 'salary' or 'commission.'",
        "grounding": "Money-mule recruitment via fake job postings is a specifically named, "
                      "actively warned-about pattern (UK NCA, Europol both run public "
                      "awareness campaigns against exactly this vector); GenAI's contribution "
                      "is generating many distinct, locally-plausible fake postings/company "
                      "fronts instead of one reused template that's easy to pattern-match.",
        "techniques": ["llm_social_engineering", "automation_at_scale"],
        "severity": "medium",
        "likelihood": "high",
    },
]

TECHNIQUE_DESCRIPTIONS = {
    "deepfake_audio": "Synthetic/cloned voice audio used for authentication bypass or impersonation.",
    "deepfake_video": "Synthetic/face-swapped video used for liveness bypass or impersonation.",
    "llm_social_engineering": "LLM-generated or LLM-adapted persuasive text/conversation.",
    "synthetic_identity": "Fabricated, internally-consistent identity used to pass verification.",
    "identity_document_forgery": "Generative-model-forged supporting documents (ID, business docs).",
    "identity_spoofing": "Impersonating a specific real entity (person or brand) rather than fabricating a new one.",
    "prompt_injection": "Malicious content designed to hijack an LLM agent's instructions/reasoning.",
    "adversarial_ml": "Techniques targeting the ML model itself (evasion, extraction, poisoning).",
    "adversarial_evasion": "Adapting attacker behavior specifically to evade a known or inferred detection rule/threshold.",
    "automation_at_scale": "Removing human-labor constraints to run many concurrent attack instances.",
}
