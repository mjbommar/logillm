"""
Demo application for email extraction tutorial.
"""

import asyncio
import os

from logillm.providers import create_provider, register_provider

from .processor import EmailProcessor

# Sample emails for testing
SAMPLE_EMAILS = [
    {
        "subject": "Order Confirmation #12345 - Your MacBook Pro is on the way!",
        "body": """Dear John Smith,

Thank you for your order! We're excited to confirm that your order #12345 has been processed.

Order Details:
- MacBook Pro 14-inch (Space Gray)
- Order Total: $2,399.00
- Estimated Delivery: December 15, 2024
- Tracking Number: 1Z999AA1234567890

If you have any questions, please contact our support team at support@techstore.com.

Best regards,
TechStore Team""",
        "sender": "orders@techstore.com",
    },
    {
        "subject": "URGENT: Server Outage - Immediate Action Required",
        "body": """Hi DevOps Team,

We're experiencing a critical server outage affecting our production environment.

Impact: All users unable to access the platform
Started: 2:30 PM EST

Please join the emergency call immediately: +1-555-123-4567

This is our highest priority.

Thanks,
Site Reliability Team""",
        "sender": "alerts@company.com",
    },
    {
        "subject": "Meeting Invitation: Q4 Planning Session",
        "body": """Hello team,

You're invited to our Q4 planning session.

When: Friday, December 20, 2024 at 2:00 PM - 4:00 PM EST
Where: Conference Room A

Please confirm your attendance by December 18th.

Best,
Sarah Johnson""",
        "sender": "sarah.johnson@company.com",
    },
    {
        "subject": "Invoice #INV-2024-001 - Payment Due",
        "body": """Dear Customer,

Your invoice for consulting services is now due.

Invoice Details:
- Invoice Number: INV-2024-001
- Amount Due: $5,250.00
- Due Date: January 15, 2025
- Payment Terms: Net 30 days

Please remit payment by the due date to avoid late fees.

Payment can be made online at portal.consulting.com or by check.

Thank you,
Accounting Department""",
        "sender": "billing@consulting.com",
    },
]


async def run_email_processing_demo() -> None:
    """Demonstration of the email processing system."""

    # Configure LogiLLM
    model = os.environ.get("MODEL", "gpt-4.1")

    if model.startswith("gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            print("⚠️  Please set OPENAI_API_KEY environment variable")
            return
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
            return
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")

    register_provider(provider, set_default=True)

    # Create our email processor
    processor = EmailProcessor()

    # Process each email and display results
    print("🚀 Email Processing Demo")
    print("=" * 50)

    results = []
    for i, email in enumerate(SAMPLE_EMAILS):
        print(f"\n📧 EMAIL {i + 1}: {email['subject'][:50]}...")

        # Process the email
        result = await processor.forward(
            email_subject=email["subject"], email_body=email["body"], sender=email["sender"]
        )

        results.append(result)

        # Display key results
        print(f"   📊 Type: {result.email_type.value}")
        print(f"   🚨 Urgency: {result.urgency.value}")
        print(f"   📝 Summary: {result.summary}")
        print(f"   🎯 Priority: {result.priority_score}/10")

        if result.financial_amount:
            print(f"   💰 Amount: ${result.financial_amount:,.2f}")

        if result.action_required:
            print("   ✅ Action Required: Yes")
            if result.deadline:
                print(f"   ⏰ Deadline: {result.deadline}")
            if result.action_items:
                print(
                    f"   📋 Actions: {', '.join(result.action_items[:2])}{'...' if len(result.action_items) > 2 else ''}"
                )
        else:
            print("   ✅ Action Required: No")

        if result.key_entities:
            print(f"   🏷️  Entities: {len(result.key_entities)} found")

    # Summary statistics
    print("\n📈 SUMMARY")
    print(f"   Total emails processed: {len(results)}")
    print(f"   Requiring action: {sum(1 for r in results if r.action_required)}")
    print(
        f"   High/Critical urgency: {sum(1 for r in results if r.urgency.value in ['high', 'critical'])}"
    )
    print(f"   Financial amounts found: {sum(1 for r in results if r.financial_amount)}")


async def main() -> None:
    """Main demo entry point."""
    await run_email_processing_demo()


if __name__ == "__main__":
    asyncio.run(main())
