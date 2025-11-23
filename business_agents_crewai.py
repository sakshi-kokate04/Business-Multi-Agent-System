# Business Multi-Agent System with OpenAI
# CrewAI Implementation

import os
from crewai import Agent, Task, Crew, Process
import streamlit as st

# Page config
st.set_page_config(page_title="Business Multi-Agent System", page_icon="🤖", layout="wide")

# Initialize environment
def setup_environment():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API Key not found! Please add it to .streamlit/secrets.toml")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key

api_key = setup_environment()

# Define Agents (they will automatically use OpenAI GPT-4)
market_research_agent = Agent(
    role='Market Research Analyst',     
    goal='Conduct comprehensive market research and identify opportunities',
    backstory="""You are an experienced market research analyst with 15 years 
    of experience in analyzing market trends, competitor strategies, and 
    identifying growth opportunities. You provide data-driven insights with 
    specific numbers and actionable recommendations.""",
    verbose=True,
    allow_delegation=False
)

sales_strategy_agent = Agent(
    role='Sales Strategy Consultant',
    goal='Develop effective sales strategies and action plans',
    backstory="""You are a seasoned sales strategist who has helped hundreds 
    of companies scale their revenue through innovative sales approaches and 
    strategic planning. You focus on practical, immediately actionable strategies 
    with clear implementation steps.""",
    verbose=True,
    allow_delegation=False
)

customer_analytics_agent = Agent(
    role='Customer Analytics Specialist',
    goal='Analyze customer behavior and generate actionable insights',
    backstory="""You are a data-driven customer analytics expert specializing 
    in customer segmentation, behavior analysis, and retention strategies. 
    You turn data into actionable business intelligence with clear metrics 
    and KPIs.""",
    verbose=True,
    allow_delegation=False
)

content_writer_agent = Agent(
    role='Business Content Writer',
    goal='Create professional business reports and documentation',
    backstory="""You are a skilled business writer who transforms complex 
    data and insights into clear, compelling reports and presentations. 
    You excel at synthesizing information from multiple sources into 
    executive-ready documents.""",
    verbose=True,
    allow_delegation=False
)

# Define Tasks
def create_market_research_task(topic):
    return Task(
        description=f"""Conduct comprehensive market research on: {topic}
        
        Your analysis must include:
        1. Market size and growth trends (with specific numbers if possible)
        2. Key competitors and their market strategies
        3. Target customer demographics and psychographics
        4. Emerging opportunities and potential threats
        5. Industry best practices and benchmarks
        
        Provide specific, actionable insights that can inform business decisions.
        Use bullet points for clarity and include concrete examples.""",
        agent=market_research_agent,
        expected_output="A detailed market research report with specific insights, data points, and actionable recommendations"
    )

def create_sales_strategy_task(topic):
    return Task(
        description=f"""Develop a comprehensive sales strategy for: {topic}
        
        Your strategy must include:
        1. Recommended sales approach and methodology
        2. Target customer segments (prioritized by potential)
        3. Pricing strategy recommendations with rationale
        4. Sales channel optimization (direct, partners, online, etc.)
        5. Key performance indicators (KPIs) and revenue targets
        6. 90-day implementation timeline with milestones
        
        Make it practical and immediately actionable with clear next steps.
        Focus on strategies that can generate quick wins.""",
        agent=sales_strategy_agent,
        expected_output="A comprehensive sales strategy document with clear action steps, timelines, and measurable goals"
    )

def create_customer_analysis_task(topic):
    return Task(
        description=f"""Analyze the customer landscape for: {topic}
        
        Your analysis must include:
        1. Customer segmentation (identify 3-5 key segments)
        2. Customer behavior patterns and purchasing trends
        3. Churn risk factors and mitigation strategies
        4. Customer lifetime value analysis framework
        5. Retention and engagement recommendations
        6. Pain points and unmet needs
        
        Provide data-backed insights with specific metrics where possible.
        Focus on actionable recommendations.""",
        agent=customer_analytics_agent,
        expected_output="A customer analytics report with detailed segmentation, insights, and retention strategies"
    )

def create_report_task(topic):
    return Task(
        description=f"""Create a professional, executive-ready business report for: {topic}
        
        The report must include:
        1. Executive Summary (3-5 key findings in bullet points)
        2. Market Analysis Overview
        3. Sales Strategy Highlights
        4. Customer Insights Summary
        5. Strategic Recommendations (prioritized by impact)
        6. Next Steps and Action Plan (30/60/90 days)
        
        Write in a clear, professional tone suitable for C-level executives.
        Use headers, bullet points, and structured formatting.
        Keep it concise but comprehensive.""",
        agent=content_writer_agent,
        expected_output="A polished business report ready for executive presentation with clear structure and actionable insights"
    )

# Streamlit App UI
def main():
    # Header
    st.title("🤖 Business Multi-Agent System")
    st.markdown("**Powered by CrewAI + OpenAI GPT-4** - Multiple AI agents working together to solve business problems")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.success("✅ OpenAI API Key Configured")
        
        st.markdown("---")
        st.header("👥 Active Agents")
        
        agents_info = [
            ("📊", "Market Research Agent", "Analyzes markets and trends"),
            ("💼", "Sales Strategy Agent", "Develops sales strategies"),
            ("👥", "Customer Analytics Agent", "Analyzes customer behavior"),
            ("📝", "Content Writer Agent", "Creates business reports")
        ]
        
        for icon, name, desc in agents_info:
            st.markdown(f"{icon} **{name}**")
            st.caption(desc)
        
        st.markdown("---")
        st.markdown("🧠 **Model:** GPT-4")
        st.info("💡 **Tip:** Be specific about your industry, product, or service for best results")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Task Configuration")
        
        task_type = st.selectbox(
            "Select Analysis Type",
            [
                "Comprehensive Business Analysis (All 4 Agents)",
                "Market Research Only",
                "Sales Strategy Only",
                "Customer Analysis Only",
                "Executive Report Only"
            ],
            help="Choose the type of analysis you need"
        )
        
        task_input = st.text_area(
            "Describe Your Business Task",
            placeholder="Example: Analyze the market opportunity for an AI-powered customer service platform targeting small e-commerce businesses in North America...",
            height=150,
            help="Be as specific as possible about your industry, target market, and business goals"
        )
        
        run_button = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Process Info")
        
        if "Comprehensive" in task_type:
            st.markdown("""
            **Analysis Steps:**
            1. 🔍 Market Research
            2. 💼 Sales Strategy
            3. 👥 Customer Analysis
            4. 📝 Final Report
            
            **Estimated Time:**
            ⏱️ 3-5 minutes
            """)
        else:
            st.markdown("""
            **Single Agent Analysis**
            
            **Estimated Time:**
            ⏱️ 45-90 seconds
            """)
        
        st.success("✅ All systems ready")
    
    # Execute Analysis
    if run_button:
        if not task_input:
            st.error("⚠️ Please enter a task description")
        else:
            with st.spinner("🤖 AI Agents are working on your analysis..."):
                try:
                    # Create tasks and agents based on selection
                    tasks = []
                    agents = []
                    
                    if "Comprehensive" in task_type:
                        # All 4 agents working sequentially
                        tasks = [
                            create_market_research_task(task_input),
                            create_sales_strategy_task(task_input),
                            create_customer_analysis_task(task_input),
                            create_report_task(task_input)
                        ]
                        agents = [
                            market_research_agent,
                            sales_strategy_agent,
                            customer_analytics_agent,
                            content_writer_agent
                        ]
                    elif "Market Research" in task_type:
                        tasks = [create_market_research_task(task_input)]
                        agents = [market_research_agent]
                    elif "Sales Strategy" in task_type:
                        tasks = [create_sales_strategy_task(task_input)]
                        agents = [sales_strategy_agent]
                    elif "Customer Analysis" in task_type:
                        tasks = [create_customer_analysis_task(task_input)]
                        agents = [customer_analytics_agent]
                    elif "Executive Report" in task_type:
                        tasks = [create_report_task(task_input)]
                        agents = [content_writer_agent]
                    
                    # Create and run crew
                    crew = Crew(
                        agents=agents,
                        tasks=tasks,
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    # Execute with status updates
                    with st.status("Running multi-agent analysis...", expanded=True) as status:
                        st.write("🔄 Initializing agents...")
                        st.write(f"📝 Processing {len(tasks)} task(s)...")
                        
                        result = crew.kickoff()
                        
                        status.update(label="✅ Analysis complete!", state="complete")
                    
                    # Display results
                    st.success("🎉 Analysis Complete!")
                    
                    st.markdown("---")
                    st.header("📄 Results")
                    
                    # Display result in expandable section
                    with st.expander("📊 Full Analysis Report", expanded=True):
                        st.markdown(result)
                    
                    # Download button
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    with col_b:
                        st.download_button(
                            label="⬇️ Download Report as Text",
                            data=str(result),
                            file_name=f"business_analysis_{task_type.lower().replace(' ', '_')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    # Success metrics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Agents Used", len(agents))
                    with col2:
                        st.metric("Tasks Completed", len(tasks))
                    with col3:
                        st.metric("Status", "✅ Success")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("💡 **Troubleshooting Tips:**")
                    st.markdown("""
                    - Verify your OpenAI API key is valid
                    - Check that you have API credits available
                    - Try with a shorter/simpler task description
                    - Ensure you have internet connectivity
                    """)
                    
                    with st.expander("🔍 See full error details"):
                        st.code(str(e))

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Built with CrewAI 🤖 | Powered by OpenAI GPT-4 | Streamlit Interface</strong></p>
        <p style='font-size: 0.9em;'>Multi-agent AI system for business analysis and strategy development</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()