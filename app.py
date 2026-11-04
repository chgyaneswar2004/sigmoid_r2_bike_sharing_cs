import os
import streamlit as st
import llm_parser as llm_parser
from nl_assistant import answer, HELP
from model_utils import WEEKDAY_NAMES

st.set_page_config(page_title="Bike demand assistant", layout="wide")
st.title("Bike demand assistant (prototype)")
st.caption("Ask about demand and bikes needed. Answers come from the demand model; every answer lists its assumptions.")

with st.sidebar:
    st.header("Planning assumptions")
    month = st.slider("Default month", 1, 12, 9, help="Used when the question names no month")
    trip = st.slider("Average trip (minutes)", 10, 60, 20, help="Assumed, not in the data. Fleet need scales with it")
    pctl = st.select_slider("Service level", [0.8, 0.9, 0.95], value=0.9, format_func=lambda x: f"P{int(x*100)}")
    st.markdown("**Example questions**\n" + HELP.split("\n", 1)[1])
    try:                                   # key from Streamlit secrets or environment, never from code
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    use_llm = st.toggle("Understand questions with Gemini", value=llm_parser.available(),
                        disabled=not llm_parser.available(),
                        help="Needs GEMINI_API_KEY. Gemini only reads the question; answers still come from the model.")
    if not llm_parser.available():
        st.caption("No GEMINI_API_KEY found: using the rule-based parser.")
settings = {"month": month, "trip_min": trip, "pctl": pctl}

if "q" not in st.session_state:
    st.session_state.q = "How many bikes do we need on Friday at 8am if it rains?"
q = st.text_input("Your question", key="q")

if q:
    r = answer(q, settings, use_llm)
    (st.error if r["status"] == "error" else st.warning if r["status"] == "refused" else st.info if r["status"] == "help" else st.success)(r["text"])
    if r["table"] is not None:
        t = r["table"].set_index("hour")
        st.line_chart(t[["forecast_rentals", "planning_rentals"]])
        st.caption(f"Question read by: {r['parser']}")
        with st.expander("Hourly details and how the question was read"):
            st.write({k: v for k, v in vars(r["query"]).items() if v not in (None, [], False)})
            st.dataframe(t)
