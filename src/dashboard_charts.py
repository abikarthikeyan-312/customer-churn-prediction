import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import load_data, get_base_metrics
import json

def get_dashboard_data(contract_filter=None, senior_filter=None):
    """Generates KPI metrics and Plotly charts as JSON strings for the Flask frontend."""
    df = load_data()
    if df is None:
        return {"success": False, "error": "No data available."}
        
    # Apply optional interactive filters
    if contract_filter and contract_filter != 'All':
        df = df[df['Contract'] == contract_filter]
        
    if senior_filter and senior_filter != 'All':
        # Telco dataset uses 0/1 for SeniorCitizen
        senior_val = 1 if senior_filter == 'Yes' else 0
        df = df[df['SeniorCitizen'] == senior_val]
        
    if len(df) == 0:
        return {"success": False, "error": "No records match the selected filters."}
        
    metrics = get_base_metrics(df)
    charts = {}
    
    # 1. Churn by Contract Type
    if 'Contract' in df.columns and 'Churn' in df.columns:
        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig1 = px.bar(contract_churn, x='Contract', y='Count', color='Churn',
                      barmode='group', color_discrete_sequence=['#3b82f6', '#ef4444'],
                      template='plotly_dark')
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=40, b=20, l=20, r=20), title="Churn by Contract Type")
        charts['contract_churn'] = json.loads(fig1.to_json())
        
    # 2. Monthly Charges Distribution
    if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
        fig2 = px.histogram(df, x='MonthlyCharges', color='Churn',
                            marginal='box', color_discrete_sequence=['#3b82f6', '#ef4444'],
                            template='plotly_dark', opacity=0.7)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=40, b=20, l=20, r=20), title="Monthly Charges Distribution")
        charts['monthly_charges'] = json.loads(fig2.to_json())
        
    # 3. Churn by Internet Service
    if 'InternetService' in df.columns and 'Churn' in df.columns:
        service_churn = df[df['Churn'] == 'Yes']['InternetService'].value_counts().reset_index()
        service_churn.columns = ['Service', 'Count']
        fig3 = px.pie(service_churn, values='Count', names='Service', hole=0.4,
                      template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=40, b=20, l=20, r=20), title="Churn by Internet Service")
        charts['internet_service'] = json.loads(fig3.to_json())
        
    # 4. Tenure vs Churn
    if 'tenure' in df.columns and 'Churn' in df.columns:
        fig4 = px.box(df, x='Churn', y='tenure', color='Churn',
                      color_discrete_sequence=['#3b82f6', '#ef4444'], template='plotly_dark')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=40, b=20, l=20, r=20), title="Tenure vs. Churn Risk")
        charts['tenure'] = json.loads(fig4.to_json())
        
    return {
        "success": True,
        "metrics": metrics,
        "charts": charts
    }
