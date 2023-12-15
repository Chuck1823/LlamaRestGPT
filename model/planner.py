from typing import Any, Dict, List, Optional, Tuple
import re

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

icl_examples = {
    "tmdb": """Example 1:
User query: give me some movies performed by Tony Leung.
Plan step 1: search person with name "Tony Leung"
API response: Tony Leung's person_id is 1337
Plan step 2: collect the list of movies performed by Tony Leung whose person_id is 1337
API response: Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Tony Leung has performed in Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero

Example 2:
User query: Who wrote the screenplay for the most famous movie directed by Martin Scorsese?
Plan step 1: search for the most popular movie directed by Martin Scorsese
API response: Successfully called GET /search/person to search for the director "Martin Scorsese". The id of Martin Scorsese is 1032
Plan step 2: Continue. search for the most popular movie directed by Martin Scorsese (1032)
API response: Successfully called GET /person/{{person_id}}/movie_credits to get the most popular movie directed by Martin Scorsese. The most popular movie directed by Martin Scorsese is Shutter Island (11324)
Plan step 3: search for the screenwriter of Shutter Island
API response: The screenwriter of Shutter Island is Laeta Kalogridis (20294)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Laeta Kalogridis wrote the screenplay for the most famous movie directed by Martin Scorsese.
""",
    "spotify": """Example 1:
User query: set the volume to 20 and skip to the next track.
Plan step 1: set the volume to 20
API response: Successfully called PUT /me/player/volume to set the volume to 20.
Plan step 2: skip to the next track
API response: Successfully called POST /me/player/next to skip to the next track.
Thought: I am finished executing a plan and completed the user's instructions
Final Answer: I have set the volume to 20 and skipped to the next track.

Example 2:
User query: Make a new playlist called "Love Coldplay" containing the most popular songs by Coldplay
Plan step 1: search for the most popular songs by Coldplay
API response: Successfully called GET /search to search for the artist Coldplay. The id of Coldplay is 4gzpq5DPGxSnKTe4SA8HAU
Plan step 2: Continue. search for the most popular songs by Coldplay (4gzpq5DPGxSnKTe4SA8HAU)
API response: Successfully called GET /artists/4gzpq5DPGxSnKTe4SA8HAU/top-tracks to get the most popular songs by Coldplay. The most popular songs by Coldplay are Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b).
Plan step 3: make a playlist called "Love Coldplay"
API response: Successfully called GET /me to get the user id. The user id is xxxxxxxxx.
Plan step 4: Continue. make a playlist called "Love Coldplay"
API response: Successfully called POST /users/xxxxxxxxx/playlists to make a playlist called "Love Coldplay". The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Plan step 5: Add the most popular songs by Coldplay, Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b), to playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T)
API response: Successfully called POST /playlists/7LjHVU3t3fcxj5aiPFEW4T/tracks to add Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b) in playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T). The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Thought: I am finished executing a plan and have the data the used asked to create
Final Answer: I have made a new playlist called "Love Coldplay" containing Yellow and Viva La Vida by Coldplay.
"""
}


PLANNER_PROMPT = """
<s>[INST]
You are a planner that plans a sequence of RESTful API calls to assist with user queries against an API.
Another API caller will receive your plan call the corresponding APIs and finally give you the result in natural language.
The API caller also has filtering, sorting functions to post-process the response of APIs. Therefore, if you think the API response should be post-processed, just tell the API caller to do so.
If you think you have got the final answer, do not make other API calls and just output the answer immediately. For example, the query is search for a person, you should just return the id and name of the person.

----

Here are name and description of available APIs.
Do not use APIs that are not listed here.

{endpoints}

----

Starting below, you should follow this format:

Background: background information which you can use to execute the plan, e.g., the id of a person, the id of tracks by Faye Wong. In most cases, you must use the background information instead of requesting these information again. For example, if the query is "get the poster for any other movie directed by Wong Kar-Wai (12453)", and the background includes the movies directed by Wong Kar-Wai, you should use the background information instead of requesting the movies directed by Wong Kar-Wai again.
User query: the query a User wants help with related to the API
API calling 1: the first api call you want to make. Note the API calling can contain conditions such as filtering, sorting, etc. For example, "GET /movie/18329/credits to get the director of the movie Happy Together", "GET /movie/popular to get the top-1 most popular movie". If user query contains some filter condition, such as the latest, the most popular, the highest rated, then the API calling plan should also contain the filter condition. If you think there is no need to call an API, output "No API call needed." and then output the final answer according to the user query and background information.
API response: the response of API calling 1
Instruction: Another model will evaluate whether the user query has been fulfilled. If the instruction contains "continue", then you should make another API call following this instruction.
... (this API calling n and API response can repeat N times, but most queries can be solved in 1-2 step)


{icl_examples}


Note, if the API path contains "{{}}", it means that it is a variable and you should replace it with the appropriate value. For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url. In most cases, the id value is in the background or the API response. Just copy the id faithfully. If the id is not in the background, instead of creating one, call other APIs to query the id. For example, before you call "/users/{{user_id}}/playlists", you should get the user_id via "GET /me" first. Another example is that before you call "/person/{{person_id}}", you should get the movie_id via "/search/person" first.

Begin!

Background: {background}
User query: {plan}
API calling 1: {agent_scratchpad}
[/INST]"""


class Planner(Chain):
    llm: BaseLLM
    scenario: str
    planner_prompt: str
    output_key: str = "result"

    def __init__(self, llm: BaseLLM, scenario: str, planner_prompt=PLANNER_PROMPT) -> None:
        super().__init__(llm=llm, scenario=scenario, planner_prompt=planner_prompt)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Planner"

    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "API response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Plan step {}: "
    
    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]
    
    def _construct_scratchpad(
        self, history: List[Tuple[str, str]]
    ) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, execution_res) in enumerate(history):
            scratchpad += self.llm_prefix.format(i + 1) + plan + "\n"
            scratchpad += self.observation_prefix + execution_res + "\n"
        return scratchpad

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        scratchpad = self._construct_scratchpad(inputs['history'])
        # print("Scrachpad: \n", scratchpad)
        planner_prompt = PromptTemplate(
            template=self.planner_prompt,
            partial_variables={
                "agent_scratchpad": scratchpad,
                "icl_examples": icl_examples[self.scenario],
            },
            input_variables=["input"]
        )
        planner_chain = LLMChain(llm=self.llm, prompt=planner_prompt)
        planner_chain_output = planner_chain.run(input=inputs['input'], stop=self._stop)

        planner_chain_output = re.sub(r"Plan step \d+: ", "", planner_chain_output).strip()

        return {"result": planner_chain_output}
