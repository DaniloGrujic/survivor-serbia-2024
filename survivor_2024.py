import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from streamlit_lottie import st_lottie
import json
import plotly.express as px
from functools import reduce
import plotly.graph_objects as go


highest_num_of_games = 0
highest_num_of_votes = 0
max_weeks = 0


class Main:
	def __init__(self):
		super().__init__()

		st.set_page_config(
			layout='wide',
			page_title="Survivor statistika",
			page_icon="logo2.png")

		path = "mucuekpQQd.json"
		with open(path, "r") as file:
			url = json.load(file)
		_, col, _ = st.columns([0.3, 0.3, 0.3])
		with col:
			st_lottie(url, height=100, width=433, speed=1, loop=True,)
		col.image("logo_final_animation.png", width=400)

		self.tab1, self.tab2, self.tab3 = st.tabs(["Plemena", "Crveni", "Plavi"])

		data = Data()

		data.define_highest_num_of_games()
		red = ['Vojislav', 'Ognjen', 'Varja', 'Mitar', 'Katarina', 'Marko', 'Tijana', 'Ivana', 'Uroš', 'Vesna', 'Stefan', 'Anđela']
		red.sort()
		for player_name in red:
			PlayerCard(self.tab2, data.get_player_data(player_name, 'Crveni'))

		blue = ['Mihaela', 'Katia', 'Luka', 'Jakob', 'Matea', 'Martina', 'Damjan', 'Nikola', 'Meggy', 'Dorian', 'Alex', 'Nancy']
		blue.sort()
		for player_name in blue:
			PlayerCard(self.tab3, data.get_player_data(player_name, 'Plavi'))

		TribeCard(self.tab1, data.tribes_data, data.get_tribes_statistics(), 'Crveno')
		TribeCard(self.tab1, data.tribes_data, data.get_tribes_statistics(), 'Plavo')

		css = '''
		<style>
			.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
			font-size:1.3rem;
			}
			#MainMenu {visibility: hidden;}
			button[title="View fullscreen"]{visibility: hidden;}
			.block-container {
			padding-top: 2rem;
			padding-bottom: 0rem;
			padding-left: 5rem;
			padding-right: 5rem;}
			a:link , a:visited{
		color: #E6B599;
		background-color: transparent;
		text-decoration: none;
		}		
		a:hover,  a:active {
		color: #e37739;
		background-color: transparent;
		text-decoration: none;
		}		
		.footer {
		position: fixed;
		top: 15;
		left: 0;
		bottom: 0;
		width: 99%;
		background-color: #0E1117;
		color: white;
		text-align: center;
		padding-top: 0.5rem;
		}
		</style>
		<div class="footer">
		<p>By: <a style=text-align: center; href="https://github.com/DaniloGrujic" target="_blank">
		<img src="https://logodix.com/logo/64439.png" alt="github" style="width:13px;height:13px;">  goofy</a></p>
		</div>
		</style>
		'''
		st.markdown(css, unsafe_allow_html=True)


class TribeCard:
	def __init__(self, parent, tribe_data, tribes_statistic, tribe_color):
		super().__init__()

		self.parent = parent

		with self.parent:
			add_vertical_space(2)

		if tribe_color == 'Crveno':
			tribe_stat = tribes_statistic['red']
			opponent_stat = tribes_statistic['blue']
		else:
			tribe_stat = tribes_statistic['blue']
			opponent_stat = tribes_statistic['red']

		tribe = tribe_data[tribe_data['Pleme'] == tribe_color]
		tribe = tribe.drop(['Pleme', 'Grad', 'Profesija'], axis=1).sort_values(by=['Status', 'Ime'], ascending=[False, True])

		col1, _, col2 = self.parent.columns([0.5, 0.1, 0.5])
		col3, col4, col5, col6 = col2.columns([0.1, 0.1, 0.1, 0.1])
		col1.subheader(f"{tribe_color} pleme", anchor=False)
		col1.dataframe(
			tribe,
			use_container_width=False,
			hide_index=True,
			width=700,
			height=457,
			column_config={
				'Godine': {'alignment': 'center'},
				'Borbe': {'alignment': 'center'},
				'Pobede': {'alignment': 'center'},
				'Brzina': {'alignment': 'center'}})

		col3.metric(
			"Pobede",
			value=tribe_stat['wins'],
			delta=tribe_stat['wins'] - opponent_stat['wins'],
			delta_color='normal',
			help='Ukupan broj osvojenih igara celog plemena')

		col4.metric(
			"Poeni",
			value=tribe_stat['points'],
			delta=int(tribe_stat['points'] - opponent_stat['points']),
			delta_color='normal',
			help='Ukupan broj osvojenih poena/borbi celog plemena')

		col5.metric(
			"Brzina",
			value=tribe_stat["speed"],
			delta=int(tribe_stat['speed'] - opponent_stat['speed']),
			delta_color='normal',
			help='Ukupan broj borbi u kojima je pleme prvo završilo "poligon"-ski deo igre, bez gadjanja')
		col6.metric(
			"Preciznost",
			value=f'{tribe_stat["accuracy"]} %',
			delta=f"{(tribe_stat['accuracy'] - opponent_stat['accuracy']).round(2)} %",
			delta_color='normal',
			help='Prosek gadjanja celog plemena')
		with col2:
			add_vertical_space(3)
		tab1, tab2, tab3, tab4 = col2.tabs(["Borbe", "Pobede", "Brzina", "Preciznost"])

		# chart 1 (games)
		stacked_h_bar_data = tribe[['Ime', 'Borbe', 'Pobede']].sort_values(by='Borbe', ascending=True)

		fig1 = go.Figure()
		fig1.add_trace(
			go.Bar(
				y=stacked_h_bar_data['Ime'],
				x=stacked_h_bar_data['Pobede'],
				name='Pobede',
				orientation='h',
				marker=dict(color="#4169E1" if tribe_color == "Plavo" else "#D22B2B"),
				hovertemplate=[
					f'{row["Ime"]}<br>Pobede: {row["Pobede"]} ({row["Pobede"]/row["Borbe"] * 100:.2f} %)<extra></extra>'
					for _, row in stacked_h_bar_data.iterrows()]

		))
		stacked_h_bar_data['Porazi'] = stacked_h_bar_data['Borbe'] - stacked_h_bar_data['Pobede']
		fig1.add_trace(
			go.Bar(
				y=stacked_h_bar_data['Ime'],
				x=stacked_h_bar_data['Porazi'],
				name='Porazi',
				orientation='h',
				marker=dict(color='#A9A9A9'),
				hovertemplate=[
					f'{row["Ime"]}<br>Porazi: {row["Porazi"]} ({row["Porazi"]/row["Borbe"] * 100:.2f} %)<extra></extra>'
					for _, row in stacked_h_bar_data.iterrows()]
		))

		fig1.update_layout(
			barmode='stack',
			width=600,
			height=280,
			margin=dict(l=50, r=50, b=10, t=30, pad=4),
			legend={'traceorder': 'normal'},
			hoverlabel=dict(bgcolor='#262730'),
			dragmode=False)

		# chart 2 (wins)
		hex_colors = {
			'Crveno': ["#300808", "#400000", "#600000", "#800000", "#A00000", "#C00000", "#E00000", "#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC"],
			'Plavo': ["#000050", "#000060", "#000080", "#0000A0", "#0000C0", "#0000E0", "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF", "#000020"]}

		pie_data = tribe[['Ime', 'Pobede']].sort_values(by='Pobede', ascending=False)

		fig2 = go.Figure()
		fig2.add_trace(
			go.Pie(
				labels=pie_data['Ime'],
				values=pie_data['Pobede'],
				hole=.3,
				marker_colors=hex_colors[tribe_color],
				hovertemplate=[f'{player}<br>Pobede: {wins}<extra></extra>' for player, wins in zip(pie_data['Ime'], pie_data['Pobede'])]))

		fig2.update_layout(
			autosize=False,
			width=500,
			height=270,
			margin=dict(l=50, r=50, b=10, t=15, pad=4),
			template='plotly_dark',
			hoverlabel=dict(bgcolor='#262730'),
			dragmode=False)

		# chart 3 (speed)
		speed = tribe_stat['weekly_speed']

		fig3 = go.Figure()
		for index, category in enumerate(speed.index):
			hover_text = [
				f"{player}<br>Nedelja: {category}<br>Brzina: {int(value)}<extra></extra>"
				for player, value in zip(speed.columns[0:], speed.loc[category])]
			fig3.add_trace(
				go.Bar(
					x=speed.columns[0:],
					y=list(speed.loc[speed.index == category][list(speed.columns[0:])].transpose().iloc[:, 0]),
					name=str(category),
					visible='legendonly' if index < speed.shape[0] - 2 else None,
					hovertemplate=hover_text))

		fig3.update_layout(
			barmode="stack",
			width=550,
			height=280,
			margin=dict(l=0, r=0, b=0, t=15, pad=4),
			legend_title="Nedelja",
			template='plotly_dark',
			hoverlabel=dict(bgcolor='#262730'),
			dragmode=False)

		# chart 4 (accuracy)
		bar_plot_data = tribe[['Ime', 'Preciznost']].sort_values(by='Preciznost', ascending=False)
		fig4 = px.bar(bar_plot_data, y='Preciznost', x='Ime', text='Preciznost')
		fig4.update_traces(
			textfont_size=12,
			textangle=0,
			textposition="outside",
			cliponaxis=False,
			marker_color="#add8e6" if tribe_color == "Plavo" else '#ff8e8e',
			hovertemplate=[
				f'{row["Ime"]}<br>Preciznost: {row["Preciznost"]}<extra></extra>'
				for index, row in bar_plot_data.iterrows()])
		fig4.update_layout(
			autosize=False,
			width=600,
			height=280,
			margin=dict(l=50, r=50, b=0, t=15, pad=4),
			hoverlabel=dict(bgcolor='#262730'),
			dragmode=False)

		for tab in tab1, tab2, tab3, tab4:
			with tab:
				add_vertical_space(1)
		tab1.plotly_chart(fig1, config={"displayModeBar": False})
		tab2.plotly_chart(fig2, config={"displayModeBar": False})
		tab3.plotly_chart(fig3, config={"displayModeBar": False})
		tab4.plotly_chart(fig4, config={"displayModeBar": False})
		self.parent.markdown("""---""")


class PlayerCard:
	def __init__(self, parent, player_data):
		super().__init__()

		self.parent = parent

		with self.parent:
			add_vertical_space(2)
		col1, col2, col3, col4, col5, col6 = self.parent.columns([0.15, 0.25, 0.15, 0.15, 0.15, 0.15])

		col1.image(f'survivor_img/{player_data["first_name"]}.jpg')
		col2.header(player_data['first_name'] + " " + player_data['last_name'], anchor=False)
		col2.subheader(f"📍{player_data['city']}, {player_data['age']}", anchor=False)
		col2.subheader(player_data['job'], anchor=False)

		col3.metric(
			"Borbe",
			value=player_data["num_games_played"],
			help="Broj borbi u kojima je takmičar učestvovao")

		col4.metric(
			"Brzina",
			value=player_data['num_races_won'],
			help='Broj borbi u kojima je takmičar prvi završio "poligon"-ski deo igre, bez gadjanja')

		col5.metric(
			"Preciznost",
			value=f'{player_data["accuracy"]} %',
			help='Broj uspešnih bacanja u odnosu na ukupan broj pokušaja')

		col6.metric("Glasovi", value=player_data['num_of_votes'], help='Ukupan broj glasova na plemenskim savetima')

		for col in [col3, col4, col5, col6]:
			with col:
				add_vertical_space(1)

		col3.metric(
			"Pobede",
			value=player_data["num_games_won"],
			delta=f"{(player_data['num_games_won'] / player_data['num_games_played'] * 100):.2f} %",
			delta_color='off',
			help='Broj osvojenih poena/borbi za svoj tim')

		col4.metric(
			"Najčešći protivnik",
			value=player_data['frequent_opponent'],
			delta=player_data['opponent_frequency'],
			delta_color='off',
			help='Protivnik sa kojim je odigrano najviše igara')

		col5.metric(
			"Najlakši protivnik",
			value=player_data['most_points_against'],
			delta=player_data['num_of_easy_point'],
			delta_color='off',
			help='Protivnik protiv kojeg takmičar ima najviše pobeda')

		col6.metric(
			"Najteži protivnik",
			value=player_data['least_points_against'],
			delta=player_data['num_of_hard_point'],
			delta_color='off',
			help='Protivnik protiv kojeg takmičar ima najviše poraza')

		with self.parent:
			add_vertical_space(1)

		col7, col8, col9 = self.parent.columns([0.3, 0.3, 0.5])

		weekly_results = player_data['weekly_results']

		col7.dataframe(
			weekly_results,
			use_container_width=False,
			hide_index=True,
			column_config={
				'Nedelja': {'alignment': 'center'},
				'Borbe': {'alignment': 'center'},
				'Pobede': {'alignment': 'center'},
				'Brzina': {'alignment': 'center'},
				'Preciznost': {'alignment': 'center'}})

		radar_df = pd.DataFrame(dict(
			r=[
				player_data['num_games_played'] / highest_num_of_games * 10,
				player_data['num_games_won'] / player_data['num_games_played'] * 10,
				player_data['speed'] * 10,
				player_data['accuracy'] / 10,
				abs(player_data['num_of_votes'] / highest_num_of_votes * 10 - 10)],
			theta=['Borbe', 'Pobede', 'Brzina', 'Preciznost', 'Glasovi']))

		fig1 = px.line_polar(
			radar_df,
			r='r',
			theta='theta',
			hover_name='r',
			line_close=True,
			template='plotly_dark',
			height=350,
			width=350,
			range_r=[0, 10],
			color_discrete_sequence=["#4169E1" if player_data['tribe_id'] == 2 else "#D22B2B"])

		fig1.update_traces(fill='toself',  hovertemplate='%{theta}: %{r:.2f}<extra></extra>', mode='markers+lines')

		fig1.update_polars(
			bgcolor="#262730",
			angularaxis_showgrid=False,
			radialaxis_gridwidth=0,
			radialaxis_tickmode='array',
			gridshape='linear',
			radialaxis_showticklabels=False)

		fig1.update_layout(
			height=280,
			margin=dict(b=10),
			polar=dict(
				radialaxis=dict(
					visible=True,
					range=[0, 10])),
			showlegend=False,
			hoverlabel=dict(bgcolor='#262730'),
			dragmode=False)

		# fig1.for_each_trace(lambda t: t.update(hoveron='points'))
		col8.plotly_chart(fig1, config={'displayModeBar': False})

		weekly_results['Preciznost'] = weekly_results['Preciznost'].apply(lambda acc_str: float(acc_str[:-1]) / 10)

		graph_colors = {1: ["#D22B2B", "#F88379", "orange", "#FFE5B4"], 2: ["#4169E1", "#89CFF0", "orange", "#FFE5B4"]}

		weeks = list(range(1, max_weeks + 1))
		x_values = weekly_results['Nedelja'] if 'Nedelja' in weekly_results else weeks

		traces = []
		for index, category in enumerate(['Borbe', 'Pobede', 'Brzina', 'Preciznost']):
			y_values = weekly_results[category] if category in weekly_results else [None] * len(x_values)
			traces.append(
				go.Bar(
					name=category,
					x=x_values,
					y=y_values,
					marker_color=graph_colors[player_data['tribe_id']][index],
					hovertemplate=[
						f'{player_data["first_name"]}<br>Nedelja: {x}<br>{category}: {y:.2f}<extra></extra>'
						for x, y in zip(x_values, y_values)]))

		fig2 = go.Figure(data=traces)

		fig2.update_layout(
			barmode='group',
			autosize=False,
			width=600,
			height=300,
			margin=dict(l=50, r=50, b=10, t=0, pad=4),
			template='plotly_dark',
			xaxis=dict(title_text="Nedelje", tickmode='array', tickvals=weeks, range=[0.5, max(weeks) + 0.5]),
			yaxis=dict(range=[0, 10]),
			hoverlabel=dict(bgcolor='#262730'),
			dragmode=False)

		col9.plotly_chart(fig2, config={'displayModeBar': False})
		self.parent.markdown("""---""")


class Data:
	def __init__(self):
		super().__init__()

		self.tribes_data = pd.read_csv("data/competitors.csv")
		self.games_data = pd.read_csv("data/game_results.csv")
		self.voting_data = pd.read_csv("data/voting.csv")

		self.tribes_data['Borbe'] = 0
		self.tribes_data['Pobede'] = 0
		self.tribes_data['Brzina'] = 0
		self.tribes_data['Preciznost'] = 0

	def get_player_data(self, player_name, player_tribe):
		player_info = self.tribes_data[self.tribes_data['Ime'] == player_name]
		player_games = self.games_data[self.games_data[player_tribe] == player_name]

		first_name = player_info.iloc[0]['Ime']
		last_name = player_info.iloc[0]['Prezime']
		city = player_info.iloc[0]['Grad']
		age = player_info.iloc[0]['Godine']
		job = player_info.iloc[0]['Profesija']

		tribe_id = 1 if player_info.iloc[0]['Pleme'] == 'Crveno' else 2
		opponent_tribe_id = 2 if player_info.iloc[0]['Pleme'] == 'Crveno' else 1
		opponent_tribe = 'Crveni' if tribe_id == 2 else 'Plavi'

		num_games_played = player_games.shape[0]
		num_games_won = player_games[player_games["Poen"] == tribe_id].shape[0]
		num_races_won = player_games[player_games["Poligon"] == tribe_id].shape[0]
		games_with_races = player_games[player_games[player_tribe] == player_name]['Poligon'].dropna().count()

		frequent_opponent = player_games[opponent_tribe].value_counts().index[0]
		opponent_frequency = int(player_games[opponent_tribe].value_counts().iloc[0])

		easy_opponent_games = player_games[player_games["Poen"] == tribe_id]
		most_points_against = easy_opponent_games[opponent_tribe].value_counts().index[0]
		num_of_easy_point = int(easy_opponent_games[opponent_tribe].value_counts().iloc[0])

		hard_opponent_games = player_games[player_games["Poen"] == opponent_tribe_id]
		least_points_against = hard_opponent_games[opponent_tribe].value_counts().index[0]
		num_of_hard_point = int(hard_opponent_games[opponent_tribe].value_counts().iloc[0])

		num_of_votes = self.voting_data[self.voting_data['Nominacija'] == player_name].shape[0]

		accuracy = player_games[f'{player_tribe}_preciznost'].dropna().apply(lambda acc_str: float(acc_str[:-1])).mean().round(2)

		weekly_games = player_games.groupby('Nedelja')['Poen'].count().reset_index()
		weekly_games.columns = ['Nedelja', 'Borbe']
		weekly_points = player_games[player_games['Poen'] == tribe_id].groupby('Nedelja')['Poen'].count().reset_index()
		weekly_speed = player_games[player_games['Poligon'] == tribe_id].groupby('Nedelja')['Poligon'].count().reset_index()
		player_games_clean = player_games.copy().dropna()
		player_games_clean[f'{player_tribe}_preciznost'] = player_games_clean[f'{player_tribe}_preciznost'].apply(lambda acc_str: float(acc_str[:-1]))
		weekly_acc = player_games_clean.groupby('Nedelja')[f'{player_tribe}_preciznost'].mean().round(1).reset_index()
		weekly_acc[f'{player_tribe}_preciznost'] = weekly_acc[f'{player_tribe}_preciznost'].apply(lambda acc_float: f'{acc_float} %')

		weekly_dfs = [weekly_games, weekly_points, weekly_speed, weekly_acc]

		weekly_results = reduce(lambda left, right: pd.merge(left, right, on='Nedelja', how='outer'), weekly_dfs)
		weekly_results.columns = ['Nedelja', 'Borbe', 'Pobede', 'Brzina', 'Preciznost']
		weekly_results.fillna(0, inplace=True)

		player_data = {
			"first_name": first_name,
			"last_name": last_name,
			"tribe_id": tribe_id,
			"city": city,
			"age": age,
			"job": job,
			"num_games_played": num_games_played,
			"num_games_won": num_games_won,
			"num_races_won": num_races_won,
			"speed": num_races_won / games_with_races,
			"opponent_frequency": opponent_frequency,
			"frequent_opponent": frequent_opponent,
			"most_points_against": most_points_against,
			"num_of_easy_point": num_of_easy_point,
			"least_points_against": least_points_against,
			"num_of_hard_point": num_of_hard_point,
			'num_of_votes': num_of_votes,
			'accuracy': accuracy,
			'weekly_results': weekly_results
		}

		self.tribes_data.set_index('Ime', inplace=True, drop=False)
		self.tribes_data.at[first_name, 'Borbe'] = num_games_played
		self.tribes_data.at[first_name, 'Pobede'] = num_games_won
		self.tribes_data.at[first_name, 'Brzina'] = num_races_won
		self.tribes_data.at[first_name, 'Preciznost'] = f'{accuracy:.2f} %'
		self.tribes_data.reset_index(drop=True, inplace=True)

		return player_data

	def define_highest_num_of_games(self):
		global highest_num_of_games, highest_num_of_votes, max_weeks

		# finding highest number of games played by one of players and setting it as a highest_num_of_games
		red_most_games_player_count = self.games_data["Crveni"].value_counts().max()
		blue_most_games_player_count = self.games_data["Plavi"].value_counts().max()
		if red_most_games_player_count >= blue_most_games_player_count:
			highest_num_of_games = red_most_games_player_count
		else:
			highest_num_of_games = blue_most_games_player_count

		# finding highest number of votes received by one of players and setting it as a highest_num_of_votes
		highest_num_of_votes = self.voting_data['Nominacija'].value_counts().max()

		# finding max number of weeks
		max_weeks = self.games_data['Nedelja'].max()

	def get_tribes_statistics(self):
		df = self.games_data.copy()
		unique_df = df.drop_duplicates(subset=['Igra', 'Nedelja', 'Runda'])

		red_wins = df[df['Poen'] == 1].groupby(['Nedelja', 'Igra'])['Poen'].count().reset_index()
		red_wins.columns = ['Nedelja', 'Igra', 'Poeni_crveni']

		blue_wins = df[df['Poen'] == 2].groupby(['Nedelja', 'Igra'])['Poen'].count().reset_index()
		blue_wins.columns = ['Nedelja', 'Igra', 'Poeni_plavi']

		tribes_combined = pd.merge(red_wins, blue_wins, on=['Nedelja', 'Igra'], how='outer')
		tribes_combined.fillna(0, inplace=True)
		tribes_combined['Wins'] = (tribes_combined['Poeni_crveni'] - tribes_combined['Poeni_plavi']).apply(lambda x: 1 if x > 0 else 2)
		red_wins = tribes_combined[tribes_combined['Wins'] == 1].shape[0]
		blue_wins = tribes_combined[tribes_combined['Wins'] == 2].shape[0]

		red_points = unique_df[unique_df['Poen'] == 1].groupby(['Nedelja', 'Igra'])['Poen'].count().reset_index()
		red_points.columns = ['Nedelja', 'Igra', 'Poeni_crveni']
		red_points = red_points['Poeni_crveni'].sum()

		blue_points = unique_df[unique_df['Poen'] == 2].groupby(['Nedelja', 'Igra'])['Poen'].count().reset_index()
		blue_points.columns = ['Nedelja', 'Igra', 'Poeni_plavi']
		blue_points = blue_points['Poeni_plavi'].sum()

		accuracy = df.dropna()
		red_accuracy = accuracy['Crveni_preciznost'].apply(lambda acc_str: float(acc_str[:-1])).mean().round(2)
		blue_accuracy = accuracy['Plavi_preciznost'].apply(lambda acc_str: float(acc_str[:-1])).mean().round(2)

		speed = df['Poligon'].dropna().value_counts()
		red_speed = speed.loc[1.0]
		blue_speed = speed.loc[2.0]

		weekly_speed_red = df[df['Poligon'] == 1.0].groupby(['Nedelja', 'Crveni'])['Poligon'].count().reset_index()
		weekly_speed_red.sort_values(by='Crveni')
		weekly_speed_red = weekly_speed_red.pivot(index='Crveni', columns='Nedelja')['Poligon'].fillna(0).rename_axis(columns=None).reset_index()
		weekly_speed_red_t = weekly_speed_red.T
		weekly_speed_red_t.columns = weekly_speed_red_t.iloc[0]
		weekly_speed_red_t = weekly_speed_red_t.drop(index=['Crveni'])
		weekly_speed_red_t['Varja'] = 0

		weekly_speed_blue = df[df['Poligon'] == 2.0].groupby(['Nedelja', 'Plavi'])['Poligon'].count().reset_index()
		weekly_speed_blue.sort_values(by='Plavi')
		weekly_speed_blue = weekly_speed_blue.pivot(index='Plavi', columns='Nedelja')['Poligon'].fillna(0).rename_axis(columns=None).reset_index()
		weekly_speed_blue_t = weekly_speed_blue.T
		weekly_speed_blue_t.columns = weekly_speed_blue_t.iloc[0]
		weekly_speed_blue_t = weekly_speed_blue_t.drop(index=['Plavi'])

		tribes_statistics = {
			'red': {
				'wins': red_wins,
				'points': red_points,
				'accuracy': red_accuracy,
				'speed': red_speed,
				'weekly_speed': weekly_speed_red_t
				},
			'blue': {
				'wins': blue_wins,
				'points': blue_points,
				'accuracy': blue_accuracy,
				'speed': blue_speed,
				'weekly_speed': weekly_speed_blue_t
				}
		}
		return tribes_statistics


if __name__ == "__main__":
	main = Main()
