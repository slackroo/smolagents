from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
import os
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
from smolagents import LiteLLMModel

from decouple import Config, RepositoryEnv

env_path = r""
env = Config(RepositoryEnv(env_path))
GEMINI_API_KEY = env('GEMINI_API_KEY')
HF_TOKEN = env('HF_TOKEN')


# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:  # it's import to specify the return type
    # Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"


def get_time_in_seconds(timezone_name: str) -> int:
    """
    Returns the current time in the given timezone as seconds since epoch.

    Args:
        timezone_name (str): Name of the timezone (e.g., 'America/New_York')

    Returns:
        int: Seconds since the Unix epoch in the given timezone.
    """
    tz = pytz.timezone(timezone_name)
    now = datetime.datetime.now(tz)
    # Convert to UTC then to timestamp (seconds since epoch)
    return int(now.astimezone(pytz.utc).timestamp())


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool
def calculate_time_difference(timezone: str, other_timezone: str) -> float:
    """Fetches the time difference between two timezones in hours.

    Args:
        timezone: the reference timezone (e.g., "Asia/Kolkata")
        other_timezone: the other timezone to compare with (e.g., "Europe/London")

    Returns:
        float: The time difference between the two timezones in hours.
    """
    try:
        # Create timezone objects
        tz1 = pytz.timezone(timezone)
        tz2 = pytz.timezone(other_timezone)

        # Get the current time in each timezone
        # now = datetime.datetime.utcnow()
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        # Current UTC time
        time1_local = tz1.normalize(utc_now.astimezone(tz1))
        time2_local = tz2.normalize(utc_now.astimezone(tz2))

        # Compute the total difference in seconds
        time_diff_seconds = (time1_local.utcoffset().total_seconds() -
                             time2_local.utcoffset().total_seconds())

        # Convert seconds to hours
        return time_diff_seconds / 3600
    except Exception as e:
        raise ValueError(f"Error calculating time difference: {str(e)}")


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

# model = HfApiModel(
# max_tokens=2096,
# temperature=0.5,
# # model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',
# token=
# custom_role_conversions=None,
# )

# model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-thinking-exp-1219", api_key=GEMINI_API_KEY)
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-lite", api_key=GEMINI_API_KEY)

# Import tool from Hub
image_generator = load_tool("agents-course/text-to-image", token=HF_TOKEN , trust_remote_code=True)
search_tool = DuckDuckGoSearchTool()

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, calculate_time_difference, image_generator, search_tool],
    ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()
