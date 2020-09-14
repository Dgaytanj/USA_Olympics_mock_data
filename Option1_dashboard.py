#!/usr/bin/env python
# coding: utf-8

# In[1]:


import panel as pn
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import datetime as dt
from datetime import timedelta

from scipy.stats.kde import gaussian_kde
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pn.extension('plotly')


# In[2]:


#Survey_Completed timestamp update
def adjusted_date(date_submit, entry_date, true_date):
    if date_submit.year < 2020:
        if entry_date == 1:
            return date_submit
        if entry_date == 2:
            return (date_submit - timedelta(days=1))
        if entry_date == 3:
            return date_submit - timedelta(days=2) 
        if entry_date == 4:
            return date_submit - timedelta(days=3) 
        if entry_date == 5:
            if true_date == 'today':
                return date_submit
            else:
                return true_date
    else:
        if true_date == 'today':
                return date_submit
        else:
            return true_date
#Exponentially weighted moving average acute:chronic workload ratio       
def EWMA_Acute(df,acute_duration):
    EWMA_AcuteWR_val = []
    lambda_a = 2/(acute_duration+1)
    for ind in df.index:
        if ind == df.index.min():
            EWMA_AcuteWR_val.append(df['Total_Load'][ind])
        else:
            EWMA_AcuteWR_val.append((df['Total_Load'][ind]*lambda_a)+((1-lambda_a)*(EWMA_AcuteWR_val[ind-1])))
    return EWMA_AcuteWR_val
            
def EWMA_Chronic(df,chronic_duration):
    EWMA_ChronicWR_val = []
    lambda_a = 2/(chronic_duration+1)
    for ind in range(len(df)):
        if ind == df.index.min():
            EWMA_ChronicWR_val.append(df['Total_Load'][ind])
        else:
            EWMA_ChronicWR_val.append((df['Total_Load'][ind]*lambda_a)+((1-lambda_a)*(EWMA_ChronicWR_val[ind-1])))
    return EWMA_ChronicWR_val


# In[3]:


# Update folder Path below:
PATH = "//Users//DanTheManWithAPlan//Documents//USOPC_MockData_Daniel_Gaytan_Jenkins//" ##UPDATE
load_df = pd.read_csv(PATH+'Mock_Load_Data.csv',engine='python')
travel_df = pd.read_csv(PATH+'Mock_Travel_Location_Data.csv',engine='python')
wellness_df = pd.read_csv(PATH+'Mock_Wellness_Data.csv',engine='python')


# In[4]:


# Cleaning and formating data:
load_df['Date_New'] = load_df['Date_New'].replace({'0':np.nan, 0:np.nan})
load_df['Survey_Completed'] = pd.to_datetime(load_df.Survey_Completed)
load_df['Athlete'] = load_df['Athlete'].str.replace('Athlete ', '').astype(int)
load_df['TrainingY/N'].fillna('N/A', inplace=True)
load_df['SameInjury'].fillna('N/A', inplace=True)
load_df['InjuryFirstTimeReport'].fillna('N/A', inplace=True)
load_df['Illness/InjuryReport'].fillna('N/A', inplace=True)
load_df['IllnessFirstTimeReport'].fillna('N/A', inplace=True)
load_df['Illness/InjuryReport'] = load_df['Illness/InjuryReport'].replace('Yes, I have an�injury to report','Yes, I have an injury to report')


travel_df['Date'] = pd.to_datetime(travel_df.Date)
travel_df['Athlete'] = travel_df['Athlete'].str.replace('Athlete ', '').astype(int)

wellness_df['Survey_Completed'] = pd.to_datetime(wellness_df.Survey_Completed)
wellness_df['Athlete'] = wellness_df['Athlete'].str.replace('Athlete ', '').astype(int)

load_df['Date'] = pd.to_datetime(load_df.apply(lambda x: adjusted_date(x.Survey_Completed, x.DayOfEntry, x.Date_New), axis=1))
wellness_df['Date'] = pd.to_datetime(wellness_df.apply(lambda x: adjusted_date(x.Survey_Completed, x.DayOfEntry, x.Date_New), axis=1))


# In[5]:


#Player load metrics
player_load = {}

for player in load_df.Athlete.unique():
    athlete_load = load_df[load_df['Athlete'] == player].set_index(pd.DatetimeIndex(load_df[load_df['Athlete'] == player]['Date']).normalize())    
    resample_load = athlete_load[['Practice_Load']].resample('D').sum()
    resample_df = resample_load.join(athlete_load, how = 'left',lsuffix='drop').drop(['Practice_Loaddrop','Date',
                                                                                      'Survey_Completed','Date_New',
                                                                                      'DayOfEntry','DayOfEntry_TEXT'],axis=1).reset_index()
    col = resample_df.columns[resample_df.dtypes == 'float64'].values
    resample_df[col] = resample_df[col].fillna(0)
    resample_df['Athlete'] = resample_df['Athlete'].fillna(resample_df['Athlete'].unique()[0])
    resample_df['Team'] = resample_df['Team'].fillna(resample_df['Team'].unique()[0])
    #resample_df = resample_df.drop_duplicates(subset='Date', keep="last")
    player_load[player] = resample_df.drop_duplicates(subset='Date', keep="last").reset_index(drop=True)
    player_load[player]['Total_Load'] = player_load[player]['Practice_Load']+player_load[player]['Strength_Load']+player_load[player]['Conditioning_Load']+player_load[player]['Competition_Load']
    player_load[player]['Acute_Workload'] = EWMA_Acute(player_load[player],7)
    player_load[player]['Chronic_Workload'] = EWMA_Acute(player_load[player],28)
    player_load[player]['EWMA_ACWR'] = player_load[player]['Acute_Workload']/player_load[player]['Chronic_Workload']
    player_load[player]['EWMA_ACWR'] = player_load[player]['EWMA_ACWR'].fillna(0)


# In[6]:


#Player wellness formating
player_wellness = {}

for player in wellness_df.Athlete.unique():
    athlete_wellness = wellness_df[wellness_df['Athlete'] == player].set_index(pd.DatetimeIndex(wellness_df[wellness_df['Athlete'] == player]['Date']).normalize())    
    resample_wellness = athlete_wellness[['Sleep_Quality']].resample('D').sum()
    resample_wellness_df = resample_wellness.join(athlete_wellness, how = 'left',lsuffix='drop').drop(['Sleep_Qualitydrop','Date',
                                                                                                      'Survey_Completed','Date_New',
                                                                                                      'DayOfEntry','DayOfEntry_TEXT'],axis=1).reset_index()
    col = resample_wellness_df.columns[resample_wellness_df.dtypes == 'float64'].values
    resample_wellness_df[col] = resample_wellness_df[col].fillna(0)
    resample_wellness_df['Athlete'] = resample_wellness_df['Athlete'].fillna(resample_wellness_df['Athlete'].unique()[0])
    resample_wellness_df['Team'] = resample_wellness_df['Team'].fillna(resample_wellness_df['Team'].unique()[0])
    player_wellness[player] = resample_wellness_df.drop_duplicates(subset='Date', keep="last").reset_index(drop=True)
    


# In[137]:


import panel as pn
import datetime as dt
pn.extension()



#DiscreteSlider widget
athlete_selector = pn.widgets.Select(
                                          options=list(sorted(player_load.keys())),
                                          value=1,
                                          margin = (-10,30,10,10))
#DatePicker_DailyWellness
DailyWellness_date_picker = pn.widgets.DatePicker(
                                                 value = (player_load[athlete_selector.value]['Date'].max()).to_pydatetime().date(),
                                                 margin = (-10,10,10,10))
#DateRangeSlider widget
ACWR_DateRangeSlider = pn.widgets.DateRangeSlider(name='Date Range',
                                               start=(player_load[athlete_selector.value]['Date'].min()).to_pydatetime(),
                                               end=(player_load[athlete_selector.value]['Date'].max()).to_pydatetime(),
                                               value=((player_load[athlete_selector.value]['Date'].min()).to_pydatetime(),
                                                      (player_load[athlete_selector.value]['Date'].max()).to_pydatetime()),
                                                 bar_color = '#000f8a',
                                                 )

#Attempting to update the year of the DateRangeSlider when pn.DiscreteSlider.value is changed
def callback(event):
    ACWR_DateRangeSlider.start = (player_load[event.new]['Date'].min()).to_pydatetime()
    ACWR_DateRangeSlider.end = (player_load[event.new]['Date'].max()).to_pydatetime()                                      
    ACWR_DateRangeSlider.value = (ACWR_DateRangeSlider.start,ACWR_DateRangeSlider.end)
    DailyWellness_date_picker.value = (player_load[event.new]['Date'].max()).to_pydatetime().date()

@pn.depends(athlete_selector.param.value,DailyWellness_date_picker.param.value)
def sleep_status(player,current_date):
    colors = ['royalblue', 'aquamarine']
    data_df = player_wellness[player][player_wellness[player]['Date'].dt.date == current_date].reset_index(drop=True)
    labels = ['Sleep Hours','Nap Hours']
    values = data_df[['Sleep_Hours','Nap_Hours']].values.tolist()[0]
    sleep_quality = 'Sleep Quality<br>'+str(data_df['Sleep_Quality'].values.tolist()[0])+'/100'
    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig.update_traces(hoverinfo='percent+label', textinfo='value+label', textfont_size=16,
                      textposition='inside',
                     marker=dict(colors=colors))
    fig.layout.autosize = True
    fig.update_layout(template="plotly_dark",
                      title={'text': "Sleep Status",
                            'y':0.94,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      font=dict(size=14,
                                color="white"),
                     annotations=[dict(text=sleep_quality, x=0.50, y=0.5, font_size=18, showarrow=False)],
                     legend=dict(orientation="h",yanchor="bottom",y=-0.1,xanchor="left",x=0.2,
                                 font=dict(size=16,
                                            color="white")),
                      margin=dict(l=10, r=10, t=60, b=10),
                     #autosize=False,
                      #width=485,
                      #height=450,
                     #paper_bgcolor='#282828',plot_bgcolor='#282828'
                     )
    return pn.pane.Plotly(fig,config={'responsive': True},
                         width = 500,
                         height = 520,
                         sizing_mode='stretch_width')

@pn.depends(athlete_selector.param.value,DailyWellness_date_picker.param.value)
def wellness_status(player,current_date):
    data_df = player_wellness[player][player_wellness[player]['Date'].dt.date == current_date].reset_index(drop=True)
    labels = ['Readiness','Motivation','Fatigue','Soreness','Stress']
    neg_values = data_df[['Fatigue','Soreness','Stress']].values.tolist()[0]
    pos_values = data_df[['Readiness','Motivation']].values.tolist()[0]
    colors = []
    for val in pos_values:
        if val <= 33:
            colors.append('#9b0000')
        elif val > 33 and val <= 66:
            colors.append('#fcc100')
        elif val > 66:
            colors.append('#1d9200')
    for val in neg_values:
        if val <= 33:
            colors.append('#1d9200')
        elif val > 33 and val <= 66:
            colors.append('#fcc100')
        elif val > 66:
            colors.append('#9b0000')

    values = pos_values+neg_values

    fig = go.Figure(go.Bar(x=values,
                           y=labels,
                           marker={'color': colors},
                           orientation='h',
                          hovertemplate = "%{x}/100<extra></extra>"))
    
    fig.update_xaxes(nticks=9,tickfont = dict(size = 15,color = 'white'))
    fig.update_yaxes(tickfont = dict(size = 15,color = 'white'))
    fig.layout.autosize = True
    fig.update_layout(template="plotly_dark",
                      title={'text': "Wellness Status",
                            'y':0.94,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      xaxis_title="Reporting Scale (0-100)",
                      font=dict(size=14,
                                color="white"),
                      margin=dict(l=10, r=10, t=60, b=10),
                      #autosize=False,width=485,height=450,
                      #paper_bgcolor='#282828',plot_bgcolor='#282828'
                     )
    return pn.pane.Plotly(fig,config={'responsive': True},
                         width = 500,
                         height = 520,
                         sizing_mode='stretch_width')

@pn.depends(athlete_selector.param.value,ACWR_DateRangeSlider.param.value)
def workload_plot(player,start_end_date):
    df = player_load[player]
    cut_load_df = df[(df['Date'] >= start_end_date[0])&(df['Date'] <= start_end_date[1])]
    variables = ['Practice_Load','Strength_Load','Conditioning_Load','Competition_Load']
    compiled_load = []
    for var in variables:
        single_load_type = pd.DataFrame(cut_load_df[['Date',var]])
        single_load_type['Load_Type'] = var.replace('_', ' ')
        compiled_load.extend(single_load_type.values.tolist())
        load_plot_df = pd.DataFrame(compiled_load, columns=['Date','Load','Load_Type'])
    
    # Create figure with secondary y-axis
    fig_load = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig_load.add_trace(
        go.Bar(name='Practice', x=load_plot_df[load_plot_df['Load_Type'] == 'Practice Load']['Date'].dt.date.tolist(), 
               y=load_plot_df[load_plot_df['Load_Type'] == 'Practice Load']['Load'].tolist(),
              hovertemplate = "<b>Date</b>: %{x}"+"<br><b>Load</b>: %{y}"),
        secondary_y=False)

    fig_load.add_trace(
        go.Bar(name='Strength', x=load_plot_df[load_plot_df['Load_Type'] == 'Strength Load']['Date'].dt.date.tolist(), 
               y=load_plot_df[load_plot_df['Load_Type'] == 'Strength Load']['Load'].tolist(),
              hovertemplate = "<b>Date</b>: %{x}"+"<br><b>Load</b>: %{y}"),
        secondary_y=False)

    fig_load.add_trace(
        go.Bar(name='Conditioning', x=load_plot_df[load_plot_df['Load_Type'] == 'Conditioning Load']['Date'].dt.date.tolist(), 
               y=load_plot_df[load_plot_df['Load_Type'] == 'Conditioning Load']['Load'].tolist(),
              hovertemplate = "<b>Date</b>: %{x}"+"<br><b>Load</b>: %{y}"),
        secondary_y=False)

    fig_load.add_trace(  
        go.Bar(name='Competition', x=load_plot_df[load_plot_df['Load_Type'] == 'Competition Load']['Date'].dt.date.tolist(), 
               y=load_plot_df[load_plot_df['Load_Type'] == 'Competition Load']['Load'].tolist(),
              hovertemplate = "<b>Date</b>: %{x}"+"<br><b>Load</b>: %{y}"),
        secondary_y=False)

    fig_load.add_trace(
        go.Scatter(x=cut_load_df['Date'], y=cut_load_df['EWMA_ACWR'], name="EWMA",
                                    #text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                    #hoverinfo='text+name',
                                    mode='lines+markers',
                                    line_shape='spline',
                  hovertemplate = "<b>Date</b>: %{x}"+"<br><b>ACWR</b>: %{y}"),
        secondary_y=True)

    fig_load.update_layout(template="plotly_dark",
                           title={'text': "Acute:Chronic Workload Ratio - Exponentially Weighted Moving Average (EWMA) Model",
                            'y':0.94,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      font=dict(size=14,
                                color="white"),
                      barmode='stack',
                      #autosize=True,
                      #width=2000,
                      #height=1000,
                     #margin=dict(l=100, r=100, t=60, b=80)
                     )
    # Set x-axis title
    fig_load.update_xaxes(title_text="Date")
    fig_load.layout.autosize = True
    # Set y-axes titles
    fig_load.update_yaxes(title_text="Total Load", secondary_y=False)
    fig_load.update_yaxes(title_text="Acute:Chronic Workload Ratio (ACWR)", secondary_y=True)
    #fig_load.layout.autosize = True
    return pn.pane.Plotly(fig_load,config={'responsive': True},
                          height=600,
                          width=2000,
                          #height_policy='fit',
                          sizing_mode='stretch_width')

watcher = athlete_selector.param.watch(callback,'value')


# In[138]:


css = '''
.panel-widget-box {
  background: #f0f0f0;
  border-radius: 5px;
  border: 1px black solid;
}
'''
pn.config.raw_css.append(css)
import plotly.io as pio
pio.renderers.default = "browser"
dashboard_background = """
The Athlete Status Dashboard allows stakeholders 
to quickly review an athlete's latest self-reported 
sleep, wellness, and load data metrics. Simply  
select an athlete and filter based on the available 
date filters."""



athlete_select_info = """Select an athlete based on numerical identifier."""
wellness_date = """Adjust sleep and wellness visuals based on date. """

pn.config.sizing_mode='stretch_width'
header = pn.Row(
            #pn.layout.Spacer(width=200),
            #pn.layout.HSpacer(),
            #pn.Column(
                #pn.layout.Spacer(height=15),
                pn.pane.Str('ATHLETE STATUS REPORT', style={'font-family': "Arial Black",
                                                                   'color':'black',
                                                                   'font-size': '60px',
                                                                  },
                                 margin = (-30, 50, 0, 50),
                                 #width_policy='max',
                                 #sizing_mode='stretch_width',
                                
                                #),
            ),
            pn.layout.HSpacer(),
    pn.layout.HSpacer(),
            pn.pane.PNG(PATH+'logo-usopc-black-semi-trans.png',height=150,
                        margin = (0, 10, 0, 100),
                       ),
    #pn.layout.HSpacer(),
    background="#c30000",height=150,height_policy='fixed'
)
x = pn.Column(
    pn.Spacer(height=0, margin=0),
    header,
    pn.Spacer(height=5),
    pn.Row(
        pn.Column(
            pn.WidgetBox(
                pn.Column(
                    '## Dashboard Summary',
                    pn.layout.Divider(margin=(-20, 10, -20, 10)),
                    pn.pane.Str(dashboard_background,style={'font-size': '12pt'},margin=(-20, 0, 10, 10)),
                    background='whitesmoke'
                )),
            pn.WidgetBox(
                pn.Column('## Athlete Filters:',
                          pn.layout.Divider(margin=(-20, 10, -20, 10)),
                          pn.pane.Str(athlete_select_info,style={'font-size': '12pt'},margin=(-20, 0, 10, 10)),
                          athlete_selector,
                          background='whitesmoke')),
            pn.WidgetBox('## Sleep & Wellness Date Filter:',
                         pn.layout.Divider(margin=(-20, 10, -20, 10)),
                         pn.pane.Str(wellness_date,style={'font-size': '12pt'},margin=(-20, 0, 10, 10)),
                         DailyWellness_date_picker,background='whitesmoke'),
        width=525,width_policy='fixed',
        sizing_mode='stretch_height'),
        pn.WidgetBox(sleep_status,background = '#454545'),
         pn.WidgetBox(wellness_status,background = '#454545')
    ),
    '## Athlete Load',
    pn.layout.Divider(margin=(-20, 0, 0, 0)),
    (ACWR_DateRangeSlider),
    pn.WidgetBox(workload_plot,background = '#454545')
     )
x.margin = (0, 100, 0, 100)
x.show()


# In[ ]:




