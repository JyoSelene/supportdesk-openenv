"""
Synthetic email dataset for SupportDesk-OpenEnv.
Contains three pools of emails, one per task, with ground-truth labels.
"""
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# TASK 1 — email_classify
# Agent reads each email and must return priority + category.
# Ground truth: priority in {urgent, high, normal, low},
#               category in {billing, technical, account, general, complaint}
# ---------------------------------------------------------------------------

CLASSIFY_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "cls_001",
        "subject": "URGENT: Production API is completely down — losing $50k/hour",
        "body": (
            "Hi Support,\n\n"
            "Our entire production environment has been down for the past 2 hours. "
            "All API calls are returning 503. We've already lost an estimated $50,000 in revenue "
            "and our SLA with our own clients is breached.\n\n"
            "Stack trace attached. We need an engineer on this NOW.\n\n"
            "Account: ENT-99021\n"
            "Contact: cto@acmecorp.com\n\n"
            "— Marcus (CTO, Acme Corp)"
        ),
        "sender": "Marcus Chen",
        "sender_email": "cto@acmecorp.com",
        "timestamp": "2024-01-15T08:03:00Z",
        "thread_length": 1,
        "attachments": ["stack_trace.log", "monitoring_dashboard.png"],
        "labels": {"priority": "urgent", "category": "technical"},
    },
    {
        "email_id": "cls_002",
        "subject": "I was charged twice this month — please refund ASAP",
        "body": (
            "Hello,\n\n"
            "I just checked my bank statement and I've been charged $49.99 twice for February. "
            "This is not the first time this has happened. I want a full refund for the duplicate "
            "charge AND a credit for the inconvenience. If this isn't resolved by end of day I will "
            "be disputing both charges with my bank.\n\n"
            "Customer ID: CUS-78234\n"
            "Card ending: 4821\n\n"
            "— Lisa Monroe"
        ),
        "sender": "Lisa Monroe",
        "sender_email": "lisa.monroe@gmail.com",
        "timestamp": "2024-01-15T09:15:00Z",
        "thread_length": 1,
        "attachments": ["bank_statement.pdf"],
        "labels": {"priority": "urgent", "category": "billing"},
    },
    {
        "email_id": "cls_003",
        "subject": "Cannot log in — password reset not working",
        "body": (
            "Hi there,\n\n"
            "I've been trying to log into my account for the past hour. "
            "I clicked 'Forgot Password' but the reset email never arrived. "
            "I've checked spam too. I have a project deadline tomorrow and need access today.\n\n"
            "My account email: derek.patel@startupxyz.io\n\n"
            "Thanks,\nDerek"
        ),
        "sender": "Derek Patel",
        "sender_email": "derek.patel@startupxyz.io",
        "timestamp": "2024-01-15T10:42:00Z",
        "thread_length": 1,
        "attachments": [],
        "labels": {"priority": "high", "category": "technical"},
    },
    {
        "email_id": "cls_004",
        "subject": "Question about my invoice for December",
        "body": (
            "Hello Support Team,\n\n"
            "I received my December invoice last week and I have a question about one of the line items. "
            "It shows a charge of $15 for 'Overage fees' but I'm not sure what this covers. "
            "Could you explain what triggers this charge?\n\n"
            "Invoice number: INV-2024-0087\n\n"
            "Kind regards,\nSophia Reyes"
        ),
        "sender": "Sophia Reyes",
        "sender_email": "sophia.reyes@company.net",
        "timestamp": "2024-01-15T11:30:00Z",
        "thread_length": 1,
        "attachments": ["invoice_dec.pdf"],
        "labels": {"priority": "normal", "category": "billing"},
    },
    {
        "email_id": "cls_005",
        "subject": "What are your support hours?",
        "body": (
            "Hi,\n\n"
            "I was wondering what your customer support hours are, and whether you offer "
            "weekend support for enterprise plans.\n\n"
            "Thanks!\nAmelia Watson"
        ),
        "sender": "Amelia Watson",
        "sender_email": "amelia.w@personalemail.com",
        "timestamp": "2024-01-15T12:00:00Z",
        "thread_length": 1,
        "attachments": [],
        "labels": {"priority": "low", "category": "general"},
    },
    {
        "email_id": "cls_006",
        "subject": "Update my billing address",
        "body": (
            "Hello,\n\n"
            "I recently moved and need to update the billing address on file. "
            "Please update it to:\n\n"
            "742 Evergreen Terrace\nSpringfield, IL 62701\n\n"
            "My account email is: james.b@workemail.org\n\n"
            "Thanks,\nJames B."
        ),
        "sender": "James B.",
        "sender_email": "james.b@workemail.org",
        "timestamp": "2024-01-15T13:45:00Z",
        "thread_length": 1,
        "attachments": [],
        "labels": {"priority": "low", "category": "account"},
    },
    {
        "email_id": "cls_007",
        "subject": "Absolutely fed up — 3 outages in 2 weeks, considering cancellation",
        "body": (
            "I am writing to formally complain about the catastrophic reliability of your service. "
            "In the past two weeks we have experienced THREE separate outages totaling over 14 hours "
            "of downtime. Our team has lost work, missed deadlines, and our clients are threatening "
            "to leave us because of YOUR failures.\n\n"
            "We are an Enterprise client (Account: ENT-50312) paying $2,400/month. "
            "This is completely unacceptable. I expect a call from your VP of Engineering "
            "and a detailed post-mortem within 24 hours, or we are cancelling.\n\n"
            "— Robert Nguyen, VP Operations"
        ),
        "sender": "Robert Nguyen",
        "sender_email": "r.nguyen@bigclient.co",
        "timestamp": "2024-01-15T14:20:00Z",
        "thread_length": 3,
        "attachments": ["downtime_log.xlsx"],
        "labels": {"priority": "urgent", "category": "complaint"},
    },
    {
        "email_id": "cls_008",
        "subject": "Feature request: bulk export to CSV",
        "body": (
            "Hi team,\n\n"
            "Love the product overall! One thing that would be super helpful is a bulk export "
            "feature that lets us download our data as CSV. Right now we have to export "
            "project by project which is tedious.\n\n"
            "Would this be on the roadmap? Thanks for considering!\n\nCheers,\nKira"
        ),
        "sender": "Kira Hoffman",
        "sender_email": "kira@techstartup.io",
        "timestamp": "2024-01-15T15:00:00Z",
        "thread_length": 1,
        "attachments": [],
        "labels": {"priority": "low", "category": "general"},
    },
    {
        "email_id": "cls_009",
        "subject": "Suspicious login attempt on my account — please investigate",
        "body": (
            "Hello,\n\n"
            "I just received an alert that someone attempted to log into my account from "
            "an IP address in Eastern Europe. I did not authorize this. "
            "I have already changed my password but I'm worried my data may be at risk.\n\n"
            "Account: valerie.king@finance.org\n"
            "Alert time: 2024-01-15 07:22 UTC\n\n"
            "Please investigate and let me know if any data was accessed.\n\nValerie"
        ),
        "sender": "Valerie King",
        "sender_email": "valerie.king@finance.org",
        "timestamp": "2024-01-15T09:50:00Z",
        "thread_length": 1,
        "attachments": [],
        "labels": {"priority": "urgent", "category": "account"},
    },
    {
        "email_id": "cls_010",
        "subject": "How do I upgrade my plan?",
        "body": (
            "Hi,\n\n"
            "I'm currently on the Starter plan and I'd like to upgrade to Professional. "
            "Can you walk me through the process? Does the billing prorate automatically?\n\n"
            "Account: tommy.lee@mysite.com\n\nThanks, Tommy"
        ),
        "sender": "Tommy Lee",
        "sender_email": "tommy.lee@mysite.com",
        "timestamp": "2024-01-15T16:10:00Z",
        "thread_length": 1,
        "attachments": [],
        "labels": {"priority": "normal", "category": "account"},
    },
]


# ---------------------------------------------------------------------------
# TASK 2 — email_extract
# Agent reads each email and must extract structured fields.
# Ground truth: customer_id, issue_type, urgency_signals (list),
#               affected_product, requested_action
# ---------------------------------------------------------------------------

EXTRACT_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "ext_001",
        "subject": "Double billing on Account CUS-78234 — urgent refund request",
        "body": (
            "Hello Support,\n\n"
            "My name is Lisa Monroe, customer ID CUS-78234. I've been double-charged for my "
            "Premium Subscription this month. I saw two charges of $49.99 on my bank statement. "
            "I already tried to dispute this through the portal but it keeps timing out. "
            "I need a full refund processed by end of day or I will initiate a chargeback "
            "with my credit card company.\n\n"
            "This is extremely urgent. Please escalate.\n\nLisa Monroe"
        ),
        "sender": "Lisa Monroe",
        "sender_email": "lisa.monroe@gmail.com",
        "timestamp": "2024-01-15T09:15:00Z",
        "thread_length": 1,
        "attachments": ["bank_statement.pdf"],
        "ground_truth": {
            "customer_id": "CUS-78234",
            "issue_type": "billing_dispute",
            "urgency_signals": ["double-charged", "chargeback threat", "end of day deadline", "escalate"],
            "affected_product": "premium_subscription",
            "requested_action": "refund",
        },
    },
    {
        "email_id": "ext_002",
        "subject": "API rate limiting causing integration failures — Account ENT-99021",
        "body": (
            "To Whom It May Concern,\n\n"
            "We are enterprise account ENT-99021. Since your platform update on Jan 12th, "
            "our data pipeline integration has been hitting API rate limits that weren't "
            "present before. We're seeing HTTP 429 errors every ~15 minutes, causing our "
            "nightly ETL jobs to fail. This is impacting our Data Analytics Platform and "
            "downstream reporting.\n\n"
            "We need our rate limits reviewed and increased, or a fix for the regression. "
            "Please treat this as high priority.\n\nMarcus Chen, CTO"
        ),
        "sender": "Marcus Chen",
        "sender_email": "cto@acmecorp.com",
        "timestamp": "2024-01-15T10:30:00Z",
        "thread_length": 2,
        "attachments": ["error_logs.txt"],
        "ground_truth": {
            "customer_id": "ENT-99021",
            "issue_type": "api_rate_limit",
            "urgency_signals": ["ETL jobs failing", "high priority", "regression since update", "nightly failures"],
            "affected_product": "data_analytics_platform",
            "requested_action": "rate_limit_increase",
        },
    },
    {
        "email_id": "ext_003",
        "subject": "Cannot access team dashboard after account migration — CUS-44510",
        "body": (
            "Hi Support,\n\n"
            "We migrated from single-user to team account last Thursday (CUS-44510). "
            "Since then, three of my team members cannot access the shared dashboard. "
            "They get an 'insufficient permissions' error even though I've set them as admins. "
            "We have a client presentation using this dashboard tomorrow morning.\n\n"
            "Please help us restore access urgently.\n\n"
            "Thanks,\nNadia Okafor, Team Lead"
        ),
        "sender": "Nadia Okafor",
        "sender_email": "n.okafor@teamco.org",
        "timestamp": "2024-01-15T14:00:00Z",
        "thread_length": 1,
        "attachments": [],
        "ground_truth": {
            "customer_id": "CUS-44510",
            "issue_type": "permissions_error",
            "urgency_signals": ["client presentation tomorrow", "three users locked out", "post-migration issue"],
            "affected_product": "team_dashboard",
            "requested_action": "restore_access",
        },
    },
    {
        "email_id": "ext_004",
        "subject": "Requesting cancellation and data export — Account BIZ-30871",
        "body": (
            "Hello,\n\n"
            "I'd like to cancel my Business Plan subscription (Account BIZ-30871) effective "
            "at the end of this billing cycle (Jan 31st). Before the account closes, I need "
            "to export all project data, reports, and uploaded files. Our contract requires "
            "us to retain this data for 7 years.\n\n"
            "Please confirm the cancellation process and let me know how to initiate the bulk export.\n\n"
            "Regards,\nOliver Franks, Finance Director"
        ),
        "sender": "Oliver Franks",
        "sender_email": "o.franks@enterprise.biz",
        "timestamp": "2024-01-15T11:45:00Z",
        "thread_length": 1,
        "attachments": [],
        "ground_truth": {
            "customer_id": "BIZ-30871",
            "issue_type": "cancellation_request",
            "urgency_signals": ["end of billing cycle deadline", "compliance data retention", "Jan 31st deadline"],
            "affected_product": "business_plan_subscription",
            "requested_action": "cancel_and_export_data",
        },
    },
    {
        "email_id": "ext_005",
        "subject": "SSO integration broken after your Friday update — PRO-12209",
        "body": (
            "Team,\n\n"
            "Account PRO-12209 here. After the platform update pushed on Friday evening, "
            "our SSO via Okta is completely broken. Users are stuck in redirect loops. "
            "We have 200+ employees who cannot log in to the Project Management Suite. "
            "Our IT team verified the Okta config hasn't changed on our side.\n\n"
            "This is a blocker for our Monday stand-ups. Need an emergency fix or rollback.\n\n"
            "— IT Director, GlobalOps Inc."
        ),
        "sender": "IT Director",
        "sender_email": "it-support@globalops.com",
        "timestamp": "2024-01-13T21:00:00Z",
        "thread_length": 1,
        "attachments": ["sso_error_screenshot.png"],
        "ground_truth": {
            "customer_id": "PRO-12209",
            "issue_type": "sso_authentication_failure",
            "urgency_signals": ["200+ employees locked out", "blocker for Monday", "emergency fix needed", "post-update regression"],
            "affected_product": "project_management_suite",
            "requested_action": "fix_or_rollback",
        },
    },
    {
        "email_id": "ext_006",
        "subject": "Incorrect tax on invoices — need correction for compliance — CUS-55988",
        "body": (
            "Hello Billing Team,\n\n"
            "We recently noticed that our invoices (CUS-55988) have been applying the wrong "
            "tax rate — 8.5% instead of the correct 0% (we are a registered non-profit exempt "
            "from sales tax). This has affected the last 4 monthly invoices.\n\n"
            "We need corrected invoices issued for those months and a credit applied. "
            "We've attached our tax-exempt certificate for reference.\n\n"
            "This needs to be corrected for our financial audit next week.\n\n"
            "Best,\nAccounting, GreenEarth Foundation"
        ),
        "sender": "GreenEarth Foundation Accounting",
        "sender_email": "accounting@greenearth.org",
        "timestamp": "2024-01-15T13:00:00Z",
        "thread_length": 1,
        "attachments": ["tax_exempt_certificate.pdf"],
        "ground_truth": {
            "customer_id": "CUS-55988",
            "issue_type": "incorrect_tax_applied",
            "urgency_signals": ["financial audit next week", "4 months of incorrect invoices", "compliance issue"],
            "affected_product": "billing_invoicing",
            "requested_action": "issue_corrected_invoices_and_credit",
        },
    },
]


# ---------------------------------------------------------------------------
# TASK 3 — email_respond
# Agent reads each email + knowledge base and must compose a helpful response.
# Ground truth: required_terms (must appear in response), forbidden_phrases,
#               correct_facts (key factual details that must be accurate)
# ---------------------------------------------------------------------------

RESPOND_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "res_001",
        "subject": "Cannot log in — password reset not working",
        "body": (
            "Hi,\n\n"
            "I've been trying to log in for the past hour. I clicked 'Forgot Password' "
            "but I never received the reset email. I've checked spam. "
            "I really need access today — deadline tomorrow.\n\n"
            "Account email: derek.patel@startupxyz.io\n\nThanks, Derek"
        ),
        "sender": "Derek Patel",
        "sender_email": "derek.patel@startupxyz.io",
        "timestamp": "2024-01-15T10:42:00Z",
        "thread_length": 1,
        "attachments": [],
        "knowledge_base": {
            "password_reset_process": (
                "Password reset emails are sent from noreply@supportdesk.io and may take up to 5 minutes. "
                "Reset links expire after 24 hours. If not received, the user should: "
                "1) Check spam/junk folder, 2) Whitelist noreply@supportdesk.io, "
                "3) Try resending from the login page, 4) Contact support for a manual reset if still failing."
            ),
            "manual_reset_policy": (
                "Support agents can trigger a manual password reset from the admin panel. "
                "The user must verify identity via their registered email and account details. "
                "Manual resets are sent within 15 minutes."
            ),
        },
        "required_terms": ["noreply@supportdesk.io", "spam", "manual reset", "24 hours", "5 minutes"],
        "forbidden_phrases": ["I don't know", "I'm not sure", "cannot help"],
        "correct_facts": {
            "reset_link_expiry": "24 hours",
            "email_sender": "noreply@supportdesk.io",
            "email_delay": "5 minutes",
        },
        "response_criteria": {
            "must_acknowledge_urgency": True,
            "must_offer_manual_reset": True,
            "must_include_whitelist_tip": True,
        },
    },
    {
        "email_id": "res_002",
        "subject": "How do I upgrade from Starter to Professional plan?",
        "body": (
            "Hi,\n\n"
            "I'm on the Starter plan and want to upgrade to Professional. "
            "How does billing work? Does it prorate? Can I keep my existing data?\n\n"
            "Account: tommy.lee@mysite.com\n\nThanks, Tommy"
        ),
        "sender": "Tommy Lee",
        "sender_email": "tommy.lee@mysite.com",
        "timestamp": "2024-01-15T16:10:00Z",
        "thread_length": 1,
        "attachments": [],
        "knowledge_base": {
            "plan_upgrade_process": (
                "Users can upgrade their plan from the Account Settings > Billing page by clicking 'Upgrade Plan'. "
                "Upgrades take effect immediately. Billing is prorated: the user pays only for the remaining days "
                "in the current billing cycle at the new plan rate, then full price on renewal."
            ),
            "data_retention_on_upgrade": (
                "All existing projects, data, and settings are preserved when upgrading. "
                "New Professional plan features (advanced analytics, API access, 5 team seats) "
                "become available immediately upon upgrade."
            ),
            "professional_plan_pricing": (
                "Professional plan is $79/month or $790/year (save 17%). "
                "Starter plan is $29/month."
            ),
        },
        "required_terms": ["prorated", "Account Settings", "immediately", "data", "Professional"],
        "forbidden_phrases": ["I don't know", "not sure", "cannot confirm"],
        "correct_facts": {
            "professional_monthly_price": "$79/month",
            "billing_timing": "prorated",
            "data_preserved": True,
        },
        "response_criteria": {
            "must_explain_proration": True,
            "must_confirm_data_preserved": True,
            "must_mention_upgrade_location": True,
        },
    },
    {
        "email_id": "res_003",
        "subject": "Double billed this month — please refund",
        "body": (
            "Hello,\n\n"
            "I was charged $49.99 twice this month on my Premium subscription. "
            "Customer ID CUS-78234. I want a refund for the duplicate charge.\n\nLisa"
        ),
        "sender": "Lisa Monroe",
        "sender_email": "lisa.monroe@gmail.com",
        "timestamp": "2024-01-15T09:15:00Z",
        "thread_length": 1,
        "attachments": ["bank_statement.pdf"],
        "knowledge_base": {
            "duplicate_charge_policy": (
                "Duplicate charges are investigated within 1 business day. "
                "If confirmed, refunds are processed within 3-5 business days to the original payment method. "
                "Support agents should: 1) Apologize for the inconvenience, 2) Confirm they will escalate to billing, "
                "3) Provide the expected resolution timeline of 3-5 business days."
            ),
            "escalation_process": (
                "Billing disputes should be tagged as BILLING-ESCALATE and assigned to the Billing Team queue. "
                "Customer should receive a case number for tracking."
            ),
        },
        "required_terms": ["apologize", "refund", "3-5 business days", "billing team", "investigate"],
        "forbidden_phrases": ["cannot refund", "no refunds", "all sales final"],
        "correct_facts": {
            "refund_timeline": "3-5 business days",
            "investigation_time": "1 business day",
        },
        "response_criteria": {
            "must_apologize": True,
            "must_give_timeline": True,
            "must_offer_case_number_or_escalation": True,
        },
    },
    {
        "email_id": "res_004",
        "subject": "How do I export all my data before cancelling?",
        "body": (
            "Hello,\n\n"
            "I'm planning to cancel my Business Plan (Account BIZ-30871) at the end of this month. "
            "Before I do, I need to export all my projects, reports, and files. "
            "What's the process?\n\nOliver Franks"
        ),
        "sender": "Oliver Franks",
        "sender_email": "o.franks@enterprise.biz",
        "timestamp": "2024-01-15T11:45:00Z",
        "thread_length": 1,
        "attachments": [],
        "knowledge_base": {
            "data_export_process": (
                "Users can export all data from Account Settings > Data & Privacy > Export Data. "
                "The export includes all projects, reports, files, and account history in ZIP format. "
                "Large exports may take up to 2 hours to prepare. A download link is emailed once ready."
            ),
            "cancellation_policy": (
                "Cancellations take effect at the end of the current billing cycle. "
                "Data is retained for 30 days after cancellation for recovery purposes, then permanently deleted. "
                "Users should export data BEFORE cancelling to ensure access."
            ),
            "cancellation_process": (
                "To cancel: Account Settings > Billing > Cancel Plan. "
                "Enterprise accounts (Business and above) require a cancellation confirmation email from the account owner."
            ),
        },
        "required_terms": ["Account Settings", "export", "ZIP", "30 days", "billing cycle", "cancel"],
        "forbidden_phrases": ["data will be deleted immediately", "no export available"],
        "correct_facts": {
            "export_location": "Account Settings > Data & Privacy > Export Data",
            "data_retention_after_cancel": "30 days",
            "export_format": "ZIP",
            "export_time": "up to 2 hours",
        },
        "response_criteria": {
            "must_warn_export_before_cancel": True,
            "must_give_export_steps": True,
            "must_mention_30_day_retention": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Task configuration — maps task name to its email pool and episode setup
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "email_classify": {
        "emails": CLASSIFY_EMAILS,
        "max_steps": 10,
        "difficulty": "easy",
        "description": (
            "Read each customer email and classify its PRIORITY "
            "(urgent / high / normal / low) and CATEGORY "
            "(billing / technical / account / general / complaint)."
        ),
        "instruction_template": (
            "You are a customer support triage agent. Read the email below and classify it.\n\n"
            "EMAIL:\nFrom: {sender} <{sender_email}>\nSubject: {subject}\n\n{body}\n\n"
            "Respond with a JSON action: "
            '{{ "action_type": "classify", "priority": "<urgent|high|normal|low>", '
            '"category": "<billing|technical|account|general|complaint>" }}'
        ),
    },
    "email_extract": {
        "emails": EXTRACT_EMAILS,
        "max_steps": 6,
        "difficulty": "medium",
        "description": (
            "Read each support email and extract structured information: "
            "customer_id, issue_type, urgency_signals (list), affected_product, requested_action."
        ),
        "instruction_template": (
            "You are a support ticket extraction agent. Read the email and extract key fields.\n\n"
            "EMAIL:\nFrom: {sender} <{sender_email}>\nSubject: {subject}\n\n{body}\n\n"
            "Respond with a JSON action: "
            '{{ "action_type": "extract", "extracted_info": {{ '
            '"customer_id": "<str>", "issue_type": "<str>", '
            '"urgency_signals": ["<signal1>", ...], '
            '"affected_product": "<str>", "requested_action": "<str>" }} }}'
        ),
    },
    "email_respond": {
        "emails": RESPOND_EMAILS,
        "max_steps": 4,
        "difficulty": "hard",
        "description": (
            "Read each support email along with the provided knowledge base articles, "
            "then compose a clear, accurate, professional response to the customer."
        ),
        "instruction_template": (
            "You are a customer support agent. Read the email and use the knowledge base "
            "to write a helpful, professional response.\n\n"
            "EMAIL:\nFrom: {sender} <{sender_email}>\nSubject: {subject}\n\n{body}\n\n"
            "KNOWLEDGE BASE:\n{knowledge_base}\n\n"
            "Respond with a JSON action: "
            '{{ "action_type": "respond", "response_text": "<your full email response>" }}'
        ),
    },
}
