from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0,openai_api_key="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",llm_model_name="text-davinci-002")
#llm = OpenAI(temperature=0, model_name="text-davinci-002")
##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

#SERPAPI_API_KEY=d60ca30af2e988b9d33ece4f07102721428d32240ec631aa12b372b80943dd5d


##https://python.langchain.com/docs/modules/agents/agent_types/react

tools = load_tools(["serpapi", "llm-math"], llm=llm,serpapi_api_key="d60ca30af2e988b9d33ece4f07102721428d32240ec631aa12b372b80943dd5d")


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

#agent.run("How much did Amazon stock change in the last year? When was the last time Jeff Bezos sold Amazon stock?")

#agent.run("Why are americans travelling more this summer?")

#agent.run("What was the most significant event in the news today? and what makes this event significant? what are its implications in the near and long term??")

#agent.run("What is the most significant news today in the markets? why is this important and what does it mean for the economy in the near term")

#agent.run("Who were the biggest market movers today? what were the forces behind these big movers?")


#agent.run("Who is the largest gains for a mid-cap stock today? whats the ratio between today's gaing for this stock and the average industry's gain?")

#agent.run("Which movie has the biggest opening weekend sales in the last month? How many teenagers have watched that movie so far?")

agent.run("Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?")











