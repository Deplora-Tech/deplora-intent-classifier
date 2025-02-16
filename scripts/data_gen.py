from groq import Groq
import random
import json

client = Groq()
messages = [
        {
            "role": "user",
            "content": "I need to generate a dataset to fine tune a model for intent detection in deplora. Deplora is an intelligent deployment planning assistant. It uses generative AI to help user create, modify, and manage deployment plans and execute them to deploy software.\n\nthese are the final intents for deplora\n\ngreeting\ninsult \n\ninclude prompts like\ndeploy to aws with docker\nadd a load balancer\ndo not use multiple regions\n\n\nnow generate a dataset of 100 rows in a json file. The data should be versatile to include different user mindsets and different scenarios. be as general you can. Imagine every possible way a user can interact with the tool\n\nfollow this format\n\n{\n[\n{\"utterance\":\"change the database configuration in my deployment\",\"intent\":\"modify_deployment_plan\"},\n{\"utterance\":\"difference between AWS and Azure for deployment?\",\"intent\":\"related_question\"}\n]\n}"
        }
    ]

models = ["llama-3.3-70b-specdec", "llama3-70b-8192"]

x = 15
while True:
    try:
        model = random.choice(models)
        print(f"Generating data for model: {model}")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        res = completion.choices[0].message.content
        res_json = json.loads(res)
        res_json["model"] = model
        
        with open(f"data/abcd{x}.json", "w") as f:
            json.dump(res_json, f, indent=4)

        x += 1
    
    except Exception as e:
        print(f"Error: {e}")