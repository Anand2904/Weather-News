import asyncio
import os
from unittest import result

from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def search_weather_news(city: str) -> str:
    """Search the web for TODAY's weather news and current conditions in a city.

    Use this tool whenever the user asks about the current weather, today's
    forecast, temperature, rain, storms, heatwaves or weather alerts for a
    specific city. Do NOT use it for general knowledge, history or trivia.

    Args:
        city: Name of the city, for example "Chennai" or "London".

    Returns:
        The top web search results (title and snippet) about today's
        weather in that city, as plain text.
    """
    query = f"{city} weather today forecast news"
    try:
        with DDGS() as ddgs:
            news = ddgs.news(f"{city} weather", max_results=3)
            web = ddgs.text(query, max_results=3)
        results = list(news or []) + list(web or [])
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return f"No weather results found for {city}."

    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('body', '')}")
    return "\n".join(lines)


async def main() -> None:
    # The client reads OPENAI_API_KEY from your environment automatically
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    agent = AssistantAgent(
        name="weather_news_agent",
        model_client=model_client,
        tools=[search_weather_news],
        reflect_on_tool_use=True,
        system_message=(
            "You are a weather news assistant. When given a city, always use "
            "the search_weather_news tool to get today's weather. Then reply "
            "in exactly 3 short lines: 1) current conditions and temperature, "
            "2) today's forecast, 3) any weather alert or news. "
            "If the results lack a detail, say so briefly instead of guessing."
        ),
    )

    city = input("Enter a city: ").strip()
    if not city:
        print("Please enter a city name.")
        await model_client.close()
        return

    try:
        result = await agent.run(task=f"What is today's weather news in {city}?")
        print(result.messages[-1].content)
        # from autogen_agentchat.ui import Console
        # await Console(agent.run_stream(task=f"What is today's weather news in {city}?"))
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())