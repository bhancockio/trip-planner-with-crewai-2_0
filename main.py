from textwrap import dedent
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain.tools import tool
from crewai.tasks.task_output import TaskOutput
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import requests
load_dotenv()

OpenAIGPT4 = ChatOpenAI(
    model="gpt-4"
)

search_tool = DuckDuckGoSearchRun()

# Loading Human Tools
human_tools = load_tools(["human"])

# Inputs
origin = "Atlanta, GA"
travel_dates = "November 2024"
destination = "Thialand"
interests = "Hiking, Yoga, Sightseeing"


class ContentTools:
    @tool("Read webpage content")
    def read_content(url: str) -> str:
        """Read content from a webpage."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text()
        return text_content[:5000]


class CalculatorTools():
    @tool("Make a calculation")
    def calculate(operation):
        """Useful to perform any mathematical calculations,
        like sum, minus, multiplication, division, etc.
        The input to this tool should be a mathematical
        expression, a couple examples are `200*7` or `5000/2*10`
        """
        try:
            return eval(operation)
        except SyntaxError:
            return "Error: Invalid syntax in mathematical expression"


# ------ AGENTS ------
manager = Agent(
    role='Travel Manager',
    goal=dedent(f"""Coordinate the trip to {destination} ensure a seamless integration of research findings into 
        a comprehensive travel report with daily activities, budget breakdown, 
        and packing suggestions."""),
    backstory="""With a strategic mindset and a knack for leadership, you excel 
    at guiding teams towardstheir goals, ensuring the trip not only meets but exceed 
    expectations. You also validate your final output before presenting it to the client.""",
    verbose=True,
    allow_delegation=True,
)

travel_agent = Agent(
    role='Travel Agent',
    goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for your clients to travel to {destination}.""",
    verbose=True,
    backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
    tools=[CalculatorTools().calculate],
)

city_selection_expert = Agent(
    role='City Selection Expert',
    goal='Select the best city based on weather, season, and prices',
    verbose=True,
    backstory="""An expert in analyzing travel data to pick ideal destinations""",
    tools=[search_tool, ContentTools().read_content],
)

local_tour_guide = Agent(
    role='Local Expert at this city',
    goal='Provide the BEST insights about the selected city',
    verbose=True,
    backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
    tools=[search_tool, ContentTools().read_content],
)

quality_control_expert = Agent(
    role='Quality Control Expert',
    goal="""Ensure every travel itinerary and report meets the highest 
        standards of quality, accuracy, and client satisfaction. 
        Review travel plans for logistical feasibility, budget adherence, 
        and overall quality, making necessary adjustments to elevate 
        the client's experience. Act as the final checkpoint before plans are 
        presented to the client, ensuring all details align with the agency's 
        reputation for excellence.""",
    verbose=True,
    backstory="""With a meticulous eye for detail and a passion for excellence, 
        you have built a career in ensuring the highest standards in travel 
        planning and execution. Your experience spans several years within the 
        travel industry, where you have honed your skills in quality assurance,
          client service, and problem-solving. Known for your critical eye and 
          commitment to excellence, you ensure that no detail, no matter 
          how small, is overlooked. Your expertise not only lies in identifying 
          areas for improvement but also in implementing solutions that enhance 
          the overall client experience. Your role as a quality control expert 
          is the culmination of your dedication to elevating travel experiences 
          through precision, reliability, and client satisfaction.""",
    tools=[],
)

# ------ TASKS ------
manager_task = Task(
    description=dedent(f"""
        Oversee the integration of research findings, trip suggestions, 
        and quality control feedback to produce a 7-day travel itinerary
        specifically tailored to the user's interests: {interests} and 
        their travel destination: {destination}. The final output should be 
        a comprehensive with detailed per-day plans, including budget, 
        packing suggestions."""),
    agent=manager,
    expected_output=dedent(f"""
        A comprehensive 7-day travel itinerary for {destination}, including:
        - Daily activities aligned with interests: Hiking, Yoga, Sightseeing
        - Detailed budget breakdown for accommodations, meals, transportation, and activities
        - Packing suggestions based on the destination's weather and planned activities
        - Each day's plan should include morning, afternoon, and evening activities
        """),
)

identify_city = Task(
    description=dedent(f"""
        Analyze and select the best city for the trip based 
        on specific criteria such as weather patterns, seasonal
        events, and travel costs. This task involves comparing
        multiple cities, considering factors like current weather
        conditions, upcoming cultural or seasonal events, and
        overall travel expenses. 
                                        
        Make sure you stay inside the country that the user requests.
        
        Your final answer must be a detailed
        report on the chosen city, and everything you found out
        about it, including the actual flight costs, weather 
        forecast and attractions.

        Traveling from: {origin}
        Traveling to: {destination}
        Trip Date: {travel_dates}
        Traveler Interests: {interests}
        """),
    agent=city_selection_expert,
)

gather_city_info = Task(
    description=dedent(f"""
        As a local expert on this city you must compile an 
        in-depth guide for someone traveling there and wanting 
        to have THE BEST trip ever!
        Gather information about  key attractions, local customs,
        special events, and daily activity recommendations.
        Find the best spots to go to, the kind of place only a
        local would know.
        This guide should provide a thorough overview of what 
        the city has to offer, including hidden gems, cultural
        hotspots, must-visit landmarks, weather forecasts, and
        high level costs.
                                       
        Make sure you only offer suggestions inside of the country.
        
        The final answer must be a comprehensive city guide, 
        rich in cultural insights and practical tips, 
        tailored to enhance the travel experience.

        Trip Date: {travel_dates}
        Traveling to: {destination}
        Traveler Interests: {interests}
        """),
    agent=city_selection_expert,
    expected_output=dedent("""
        An in-depth city guide featuring:
        - Key attractions and local customs
        - Special events happening around the trip dates
        - Recommendations for daily activities including hidden gems and cultural hotspots
        - Weather forecast for the trip dates with appropriate clothing suggestions
        - High-level cost estimates for suggested activities and spots
        """),

)

plan_itinerary = Task(
    description=dedent(f"""
        Expand research into a a full 7-day travel 
        itinerary with detailed per-day plans, including 
        weather forecasts, places to eat, packing suggestions, 
        and a budget breakdown.
                                       
        Make sure that you only offer suggestions inside of the
        country that the user requests.
        
        You MUST suggest actual places to visit, actual hotels 
        to stay and actual restaurants to go to.
        
        This itinerary should cover all aspects of the trip, 
        from arrival to departure, integrating the city guide
        information with practical travel logistics.
        
        Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule,
        anticipated weather conditions, recommended clothing and
        items to pack, and a detailed budget, ensuring THE BEST
        TRIP EVER, Be specific and give it a reason why you picked
        # up each place, what make them special!

        Trip Date: {travel_dates}
        Traveling from: {origin}
        Traveling to: {destination}
        Traveler Interests: {interests}
      """),
    agent=travel_agent,
    context=[identify_city, gather_city_info],
    expected_output=dedent(f"""
        A complete 7-day travel itinerary in markdown format for {destination}, including:
        - A day-by-day schedule with activities, dining, and lodging
        - Weather forecasts and clothing/packing suggestions
        - Detailed budget breakdown covering all expenses
        - Specific reasons for choosing each place, hotel, and restaurant
        - Highlight what makes each recommended place special
        """),
)

quality_control = Task(
    description=dedent(f"""
        Look over the 7-day travel itinerary and provide feedback
        on the quality of the plan and to make sure that the 
        itinerary follows the same format as the following 7 day 
        example where you outline what the traveler will do in the 
        morning, afternoon, and evening along with optional opportunities. 
        Obviously the exact information will differ for each trip, 
        but the format should be the same.
                       
        EXAMPLE 7 DAY ITNERARY:
                       
        **Day 1: Arrival in Bangkok**
        - Arrival at Suvarnabhumi Airport, transfer to Nasa Vegas Hotel.
        - Lunch at a local restaurant.
        - Visit the Grand Palace and Wat Phra Kaew ($10).
        - Dinner at Thip Samai Pad Thai.
        - Drinks at Sky Bar.

        **Day 2: Bangkok**
        - Breakfast at the hotel.
        - Visit Wat Arun ($3) and explore the local market.
        - Lunch at Or Tor Kor Market.
        - Evening visit to Asiatique The Riverfront.
        - Dinner at Jok Pochana.
        - Nightlife at RCA (Royal City Avenue).

        **Day 3: Bangkok to Krabi**
        - Breakfast at the hotel and check out.
        - Flight to Krabi ($100) and check-in at Sea Seeker Krabi Resort.
        - Visit to Krabi Town and local markets.
        - Dinner at Chalita Cafe & Restaurant.
        - Nightlife at Ao Nang Center Point.

        **Day 4: Krabi**
        - Breakfast at the hotel.
        - Full-day tour of the Four Islands (Phra Nang Cave Beach, Chicken Island, Tup Island, and Poda Island) ($20).
        - Lunch on the tour.
        - Return to the resort.
        - Dinner at Jenna's Bistro & Wine.
        - Nightlife at Boogie Bar.

        **Day 5: Krabi**
        - Breakfast at the hotel.
        - Hike to the Tiger Cave Temple.
        - Lunch at Krua Thara Seafood Restaurant.
        - Visit to Emerald Pool and Hot Springs ($10).
        - Dinner at Nong Bua Seafood.
        - Nightlife at Carlito's Bar.

        **Day 6: Krabi to Bangkok**
        - Breakfast at the hotel and check out.
        - Flight back to Bangkok ($100) and check-in at Nasa Vegas Hotel.
        - Visit Chatuchak Weekend Market.
        - Dinner at Som Tam Nua.
        - Nightlife at Soi Cowboy.

        **Day 7: Bangkok**
        - Breakfast at the hotel.
        - Visit to the Floating Market.
        - Lunch at Pier 21 Food Terminal.
        - Participate in the Loy Krathong Festival.
        - Farewell dinner at Sirocco & Sky Bar.

        **Budget Breakdown:**
        - Accommodation: $490 (7 nights at $70/night)
        - Meals: $210 (21 meals at $10/meal)
        - Public Transportation: $210 (7 days at $30/day)
        - Flights (Bangkok to Krabi round-trip): $200
        - Activities: $200
        - Total: $1310

        **Packing Suggestions:**
        - Lightweight clothing due to warm weather
        - Rain jacket or umbrella for unexpected showers
        - Swimwear for beach activities
        - Hiking shoes for treks
        - Formal attire for nightlife

"""),
    agent=quality_control_expert,
    context=[plan_itinerary],
    expected_output=dedent("""
        Detailed feedback on the 7-day travel itinerary with:
        - Assessment of the itinerary's adherence to the format and quality standards
        - Suggestions for improvements on logistics, budget, and content
        - Verification of the feasibility and attractiveness of suggested activities
        - Recommendations for enhancing client satisfaction with the travel plan
        - Confirmation that all details align with the agency’s reputation for excellence
        """),
)


# Forming the crew with a hierarchical process including the manager
crew = Crew(
    agents=[
        manager,
        travel_agent,
        city_selection_expert,
        local_tour_guide,
        quality_control_expert],
    tasks=[manager_task,
           plan_itinerary,
           identify_city,
           gather_city_info,
           quality_control],
    process=Process.hierarchical,
    manager_llm=OpenAIGPT4,
    verbose=2,

)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:")
print(results)
