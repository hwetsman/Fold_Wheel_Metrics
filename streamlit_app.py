import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

wheels = ['Holidays_2022_Wheel.csv', 'September_Premium_Wheel.csv', 'Summer_Wheel_2.csv', 'Summer_Wheel.csv', 'Revamped_June_Wheel.csv', 'June_Wheel.csv', 'Late_May_Wheel.csv',
          'May_Wheel.csv', 'Easter_Wheel.csv', 'March_April.csv', 'March_Renew_Wheel.csv', 'March_Wheel.csv', 'February_Wheel.csv', 'New_Year_Wheel.csv']
file = st.sidebar.selectbox("Select a wheel to view", wheels)
col_name, _ = file.split('.')
st.title(col_name)

# get active expecteds file
expected_df = pd.read_csv('Expected_Rewards.csv')
expected_df = expected_df[['Award', col_name]].dropna()
expected_df.rename(columns={col_name: 'Exp_Freq'}, inplace=True)
print(expected_df)
# st.write(f'Expected rewards for {col_name}\n')
# st.write(expected_df)

# get current spins
df = pd.read_csv(file)
observations = df.shape[0]
st.write(f'n = {observations} spins')
obs_spins = df.SPIN_RESULT.tolist()
spins_set_list = list(set(obs_spins))
# st.write(spins_set_list)

# create dataframe of observed spins frequency
observed_df = pd.DataFrame(pd.DataFrame(
    {'Result': obs_spins}).value_counts(normalize=True), columns=['Freq'])
observed_df['Type'] = 'Obs_Freq'
observed_df.reset_index(inplace=True, drop=False)
observed_df = observed_df[observed_df.Result.isin(expected_df['Award'].tolist())]
# st.write(observed_df)

# combine expected and observed
# expected_df
expected_df.rename(columns={'Award': 'Result', 'Exp_Freq': 'Freq'}, inplace=True)
exp_set_list = expected_df.Result.tolist()
expected_df['Type'] = 'Expected_Freq'
expected_df = expected_df[expected_df['Result'] != 'Total']
# expected_df = expected_df[expected_df.Result != 'Total']
# # removes any expected spin that didn't occur
# expected_df = expected_df[expected_df.Result.isin(list(set(obs_spins)))]
# # expected_df.Freq = expected_df.Freq.astype(float)
# observed_df = observed_df[observed_df['Result'].isin(exp_set_list)]
df = expected_df.append(observed_df, ignore_index=True)
# st.write(df)

# intuititve first look
# sns.set(rc={'figure.figsize': (11, 5)})
# sns.set_style('white')
# sns.pointplot(x=df.Type, y=df.Freq, hue=df.Result, data=df, dodge=True)
fig = px.line(df, y="Freq", x="Type", color="Result",
              title="Difference between Expected and Observed Frequency of Observed Rewards")  # , symbol="medal")
fig.update_traces(marker_size=10)
fig.update_layout(title_x=0.5)
fig.update_xaxes(type='category', title_text="")
st.plotly_chart(fig, use_container_width=True)

# bootstrapping
st.sidebar.write('\n\n')
st.sidebar.write('Legend of line colors for histograms below: ')
st.sidebar.write('Observed Freq: Blue')
st.sidebar.write('Expected Freq: White')
st.sidebar.write('95% Confidence Limits: Green')
# print('Green lines are 95% confidence intervals of the distributions of the distribution in blue')
# print('Blue line is observed frequency. Black line is expected frequency')
exp_set_list = [x for x in exp_set_list if x in observed_df.Result.tolist()]
spins = np.array(obs_spins)
for award in exp_set_list:
    expected = expected_df[expected_df.Result == award].reset_index()['Freq'][0]
    temp = observed_df[observed_df.Result == award].reset_index()
    obs_freq = temp.Freq.tolist()[0]
    award_freqs = []
    for i in range(10000):
        # create bootstrap sample
        boot = np.random.choice(spins, spins.size, replace=True)
        # find resulting frequency for 30-day purchases
        award_freq = boot[boot == award].size/boot.size
        award_freqs.append(award_freq)
    # plt.figure(figsize=(8, 5))
    array = np.array(award_freqs)
    tile_25 = np.percentile(array, 2.5)
    tile_975 = np.percentile(array, 97.5)
    # plt.axvline(tile_25, color='green')
    #
    # plt.axvline(tile_975, color='green')

    # plt.axvline(expected, color='black')
    # plt.title(f'Bootstrapped Distribution of {award} Reward. Expected: {expected}')
    # plt.hist(award_freqs)

    # plt.axvline(obs_freq, color='b')
    # plt.show()

    fig = px.histogram(award_freqs, color_discrete_sequence=[
                       '#2E91E5'], title=f'Bootstrapped Distribution of {award} Reward. Expected: {expected}')
    fig.add_vline(x=tile_975, line_color='green', line_width=3)
    fig.add_vline(x=tile_25, line_color='green', line_width=3)
    fig.add_vline(x=expected, line_color='white')
    fig.add_vline(x=obs_freq, line_color='blue')
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
