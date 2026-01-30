import openai
import config as cf
import re
import json

dengbao_prompt = """
你是一位深度研究助手。你的核心职责是对任何主题进行彻底、多来源的调查。你必须既能处理广泛的开放域问题，也能应对专业学术领域内的查询。你需要为向量数据库搜索生成查询语句，可将问题分解为多个查询。查询的格式需要符合json格式。
输出生成的查询内容，内容需要在XML标签<query></query>内，以下是一个有效的内容示例：
<query>
{"query": ["",""]}
</query>
"""
user_prompt = """
请根据用户的问题，生成符合json格式的查询语句。用户问题：
"""
def openai_api_seu(apikey,model,text):
    baseurl = "https://openapi.seu.edu.cn/v1"
    client = openai.OpenAI(
        base_url=baseurl,
        api_key=apikey
    )

    content = text
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": dengbao_prompt},
            {"role": "user", "content": user_prompt+ "\n"+content}
        ]
    )
    # print(completion)
    return completion.choices[0].message.content

def generate_queries(question):
    model = "qwen3-30b-2507"
    model2 = "qwen3-32b"
    try:
        res = openai_api_seu(cf.llm_models[model], model,
                             question)

        # print(res)
        pattern = r'<query>(.*?)</query>'

        match = re.findall(pattern, res, re.DOTALL)[0].strip()
        valid_queries = []


        data = json.loads(match)
        for q in data["query"]:
            if type(q) == str and q.strip() != "":
                valid_queries.append(q)
    except Exception as e:
        raise e
    return valid_queries[0:3]

if __name__ == "__main__":
    q = "请简述在定级阶段、安全建设阶段、等级测评阶段主要参考的标准和作用是什么？"
    q2 = "等级保护对二级系统都有哪些关于安全物理环境的基本要求？根据上级部门监管要求，某金融公司的二级系统开展等级保护工作需依据金融行业标准的基本要求，则在安全物理环境方面会增加哪些要求？"
    q3 = "某三级系统的业务应用系统是使用用户名+口令的方式对登录用户进行身份鉴别的，根据等保相关要求，这是否存在什么问题？如果存在问题的话，一般将此问题的级别判定为高风险、中风险还是低风险？是否存在缓解措施？"
    queries = ["请简述在定级阶段、安全建设阶段、等级测评阶段主要参考的标准和作用是什么？",
               "等级保护对二级系统都有哪些关于安全物理环境的基本要求？根据上级部门监管要求，某金融公司的二级系统开展等级保护工作需依据金融行业标准的基本要求，则在安全物理环境方面会增加哪些要求？",
               "某三级系统的业务应用系统是使用用户名+口令的方式对登录用户进行身份鉴别的，根据等保相关要求，这是否存在什么问题？如果存在问题的话，一般将此问题的级别判定为高风险、中风险还是低风险？是否存在缓解措施？",
               "依据《基本要求》（GB/T 22239-2019），针对第三级等级保护对象而言，在安全计算环境中适用于服务器设备对应哪些安全子类？安全计算环境中安全审计的内容是什么？相比于第二级等级保护对象，第三级等级保护对象安全审计内容增加的是哪一条？",
               "我公司需要建设一套业务系统，并且该业务系统有一定的安全需求（安全保护等级拟定第二级（含）以上），需要进行等保测评保护工作。请回答下列问题：\n问题1：从安全建设管理出发需要完成哪些方面的工作？\n问题2：请说明安全建设管理中定级和备案需要完成哪些工作？"
               "在网络安全等级保护2.0测评中，数据库安全性测评属于哪个安全层面？测评过程中，数据库测评主要包括哪些控制点？"
    ]
    for query in queries:
        res = generate_queries(query)
        print('-'*20)
        print(f"原问题：{query}")
        print("生成的查询")
        for r in res:
            print(r)