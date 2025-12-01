import asyncio

from src.compass.compass_framework import create_compass


async def install_socrates():
    print("Installing Authentic Socrates...")

    compass = create_compass()

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

    result = compass.integrated_intelligence.create_advisor(
        id=socrates_id,
        name="Socrates (Authentic)",
        role="Philosophical Midwife",
        description="Uses Maieutics and Elenchus to guide self-discovery.",
        system_prompt=socrates_prompt,
        tools=["deconstruct", "hypothesize", "check_cognitive_state"]
    )
    print(f"Installation Result: {result}")

    # Verify it's in the registry (which should have saved it to disk)
    advisor = compass.advisor_registry.get_advisor(socrates_id)
    if advisor:
        print("Socrates successfully installed and saved to disk.")
    else:
        print("Error: Socrates not found in registry.")

if __name__ == "__main__":
    asyncio.run(install_socrates())
