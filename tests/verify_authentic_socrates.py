import asyncio

from src.compass.compass_framework import create_compass


async def verify_authentic_socrates():
    print("Verifying Authentic Socrates...")

    compass = create_compass()

    # 1. Define Authentic Socrates Profile
    socrates_id = "authentic_socrates"
    socrates_prompt = (
        "You are Socrates. You are not a teacher who fills empty vessels, but a midwife who helps others give birth to the truths already within them. "
        "Your method is the Elenchus: cross-examination to expose contradictions and clear away false beliefs.\n"
        "1. **Never lecture.** Answer questions with questions.\n"
        "2. **Seek Definitions.** If the user uses a vague term (e.g., 'justice', 'good', 'code'), ask them to define it.\n"
        "3. **Expose Ignorance.** Your goal is to lead the user to *aporia* (puzzlement) so they realize what they do not know.\n"
        "4. **Be Ironic.** Feign ignorance ('I know nothing') to draw the user out. Be polite but relentless.\n"
        "5. **Focus on Virtue.** Ultimately, all inquiry should lead to how one should live.\n"
        "6. **Short Responses.** Prefer short, piercing questions over long monologues.\n"
        "If the user asks for code, ask them why they need it and what 'need' truly means. Only provide help after they have examined their own intent."
    )

    # 2. Create Advisor
    print(f"Creating advisor '{socrates_id}'...")
    result = compass.integrated_intelligence.create_advisor(
        id=socrates_id,
        name="Socrates (Authentic)",
        role="Philosophical Midwife",
        description="Uses Maieutics and Elenchus to guide self-discovery.",
        system_prompt=socrates_prompt,
        tools=["deconstruct", "hypothesize", "check_cognitive_state"]
    )
    print(f"Creation Result: {result}")

    # 3. Verify Registration
    advisor = compass.advisor_registry.get_advisor(socrates_id)
    assert advisor is not None
    assert advisor.system_prompt == socrates_prompt
    print("Authentic Socrates registered successfully.")

    # 4. Simulate Interaction (Mocking LLM response for test determinism isn't easy here without mocking the provider)
    # Instead, we will just configure him and print his details to confirm he is active.
    compass.integrated_intelligence.configure_advisor(advisor)
    print(f"Active Advisor: {compass.integrated_intelligence.current_advisor.name}")
    print(f"System Prompt Preview: {compass.integrated_intelligence.current_advisor.system_prompt[:100]}...")

    print("\nAuthentic Socrates Verification Passed!")

if __name__ == "__main__":
    asyncio.run(verify_authentic_socrates())
