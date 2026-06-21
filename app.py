from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

app = Flask(__name__)

latest_analysis = {
    "analysis": {},
    "promotion": {}
}

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

# HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.json

        query = f"""
        Product: {data.get('product')}
        Category: {data.get('category')}
        Discount: {data.get('discount')}%
        """

        docs = vectorstore.similarity_search(
            query=query,
            k=3
        )

        catalog_context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        print("\n========== RAG CONTEXT ==========")
        print(catalog_context)
        print("=================================\n")

        final_prompt = f"""
You are Price Sense AI.

CATALOG INFORMATION:
{catalog_context}

PROMOTION DETAILS:

Product: {data.get('product')}
Category: {data.get('category')}
Discount: {data.get('discount')}%
Unit Price: {data.get('unitPrice')}
Unit Cost: {data.get('unitCost')}
Baseline Units: {data.get('baseUnits')}

Analyze the promotion.

IMPORTANT RULES:

1. Return ONLY valid JSON.
2. No markdown.
3. No explanations outside JSON.
4. All numeric values must be numbers.
5. impact must be a dollar amount string.
6. Generate a promotion score from 0-100.
7. Generate confidence percentage.
8. Explain key drivers behind recommendation.
9. Include assumptions.
10. Compare baseline vs promotion.
11. Calculate ROI percentage :- ROI = (Incremental Profit / Promotion Cost) * 100

Return this format exactly:

{{
  "verdict":"RUN",
  "headline":"",
  "summary":"",
  "promotion_score":82,
  "confidence":87,
  "metrics":{{
      "projected_lift_pct":0,
      "incremental_units":0,
      "cannibalization_pct":0,
      "net_revenue_change_pct":0,
      "net_profit_change_pct":0,
      "break_even_lift_pct":0,
      "incremental_profit":230,
      "roi_pct":18
  }},
  "recommendation_drivers":[
      ""
  ],
  "comparison":{{
      "baseline_revenue":0,
      "promo_revenue":0,
      "baseline_profit":0,
      "promo_profit":0
  }},
  "assumptions":[
    "Inventory available to support projected demand",
    "No major competitor promotion during the promotion window",
    "Price elasticity based on historical category averages",
    "Normal seasonal demand conditions"
  ],
  "cannibalization":[
      {{
          "product":"",
          "estimated_unit_decline":0,
          "impact":"-$120"
      }}
  ],
  "risks":[
      {{
          "level":"low",
          "text":""
      }}
  ],
  "alternatives":[
      {{
          "discount":15,
          "label":"15% Off",
          "verdict":"better",
          "profit":320,
          "lift":42,
          "note":""
      }},
      {{
          "discount":20,
          "label":"20% Off",
          "verdict":"better",
          "note":""
      }}
  ],
  "scenarios":{{
      "pessimistic":{{
          "lift_pct":20,
          "profit_delta":-100
      }},
      "base":{{
          "lift_pct":35,
          "profit_delta":200
      }},
      "optimistic":{{
          "lift_pct":55,
          "profit_delta":500
      }}
  }},
  "timing_note":""
}}
"""

        response = llm.invoke(final_prompt)

        text = response.content.strip()

        if text.startswith("```json"):
            text = text.replace("```json", "")

        text = text.replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            text = text[start:end]

        result = json.loads(text)

        global latest_analysis
        latest_analysis["analysis"] = result
        latest_analysis["promotion"] = data

        return jsonify(result)

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/chat", methods=["POST"])
def chat():

    try:

        question = request.json["question"]

        # Greeting shortcut
        greetings = [
            "hi",
            "hello",
            "hey",
            "start",
            "help"
        ]

        if question.lower().strip() in greetings:

            return jsonify({
                "answer": f"""👋 Promotion analyzed.

✅ Verdict: {latest_analysis['analysis'].get('verdict', 'N/A')}
📈 Sales Lift: {latest_analysis['analysis']['metrics'].get('projected_lift_pct', 0)}%
💰 Profit Change: {latest_analysis['analysis']['metrics'].get('net_profit_change_pct', 0)}%
📊 Revenue Change: {latest_analysis['analysis']['metrics'].get('net_revenue_change_pct', 0)}%

Ask me about:Ask about Risks, Recommendation, Scenarios or Alternatives.
"""
            })

        docs = vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        analysis_context = json.dumps(
            latest_analysis,
            indent=2
        )

        prompt = f"""
            You are Price Sense AI Assistant.

            Promotion Analysis:
            {analysis_context}

            Catalog Context:
            {context}

            User Question:
            {question}

            RULES:

            1. Keep answers SHORT.
            2. Maximum 5 lines.
            3. Maximum 60 words.
            4. Use simple business language.
            5. Use promotion analysis data only.
            6. No long reports.
            7. No markdown.
            8. No introductions like "Dear Team".

            Examples:

            Question: Hi

            Answer:
            👋 Promotion analyzed.

            Verdict: RUN
            Sales Lift: 35%
            Profit Change: +4.5%
            Revenue Change: +17.9%

            Question: Risks

            Answer:
            ⚠ Main risks:
            • Cannibalization from other pack sizes
            • Lower-than-expected sales lift
            • Margin pressure from discount

            Question: Explain report

            Answer:
            The promotion increases sales by 35% and revenue by 17.9%.
            Profit remains positive, so the recommendation is RUN.
            Cannibalization is limited and risk is low.

            Question: Final recommendation

            Answer:
            ✅ RUN.
            Expected sales growth exceeds break-even.
            Profit remains positive despite minor cannibalization.
            """

        response = llm.invoke(prompt)

        return jsonify({
            "answer": response.content
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

