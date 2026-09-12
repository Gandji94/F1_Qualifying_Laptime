import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_absolute_error
from typing import Sequence,Optional, Literal, Dict, Any
from dataclasses import dataclass
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

################################## Data Cleaning ##################################
def f1_rule_era(row):
    season = int(row['Season'])
    if season < 2022:
        return 'High-Downforce'
    elif season < 2026:
        return 'Ground-Effect'
    else:
        return 'Active-Aero-Hybrid-Era'
    
def abandoned_lap(df):
    if pd.isna(df['Sector1Time']) or pd.isna(df['Sector2Time']) or pd.isna(df['Sector3Time']):
        return 1
    else:
        return 0


# -----------------------------
# QUALI TIME DISTRIBUTION DIAGNOSTICS
# -----------------------------
def inspect_quali_time_distribution(df_quali, gp='Australian GP', season=2018, gap_threshold=3.0, top_n_gaps=15, w_vis = True):
    diag = df_quali.copy()

    # exact elapsed minutes from session start
    diag['Time_Minutes_Exact'] = diag['Time'].dt.total_seconds() / 60
    diag = diag.sort_values('Time_Minutes_Exact').reset_index(drop=True)
    diag['Lap_Order'] = np.arange(1, len(diag) + 1)
    diag['Time_Diff'] = diag['Time_Minutes_Exact'].diff()

    # minute-level lap counts
    diag['Minute_Bin'] = np.floor(diag['Time_Minutes_Exact']).astype(int)
    minute_counts = (
        diag.groupby('Minute_Bin')
        .size()
        .rename('Lap_Count')
        .reset_index()
    )

    # identify larger gaps
    diag['Prev_Time_Min'] = diag['Time_Minutes_Exact'].shift(1)
    gap_df = diag.loc[diag['Time_Diff'] >= gap_threshold, [
        'Lap_Order', 'Prev_Time_Min', 'Time_Minutes_Exact', 'Time_Diff'
    ]].copy()

    gap_df = gap_df.rename(columns={
        'Prev_Time_Min': 'Gap_Start_Min',
        'Time_Minutes_Exact': 'Gap_End_Min'
    })
    gap_df['Gap_Mid_Min'] = (gap_df['Gap_Start_Min'] + gap_df['Gap_End_Min']) / 2

    print("\n==============================")
    print(f"QUALI TIME DISTRIBUTION SUMMARY {gp}-{season}")
    print("==============================")
    print(f"Total laps: {len(diag)}")
    print(f"Session start minute: {diag['Time_Minutes_Exact'].min():.2f}")
    print(f"Session end minute:   {diag['Time_Minutes_Exact'].max():.2f}")
    print(f"Gap threshold used:   {gap_threshold:.2f} minutes")

    print("\nTop gaps:")
    if len(gap_df) > 0:
        print(
            gap_df.sort_values('Time_Diff', ascending=False)
                  .head(top_n_gaps)
                  .to_string(index=False)
        )
    else:
        print("No gaps found above threshold.")

    # -----------------------------
    # PLOTS
    # -----------------------------
    if w_vis:
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), constrained_layout=True)

        # 1) exact lap times over lap order
        axes[0].scatter(diag['Time_Minutes_Exact'], diag['Lap_Order'], alpha=0.7)
        axes[0].set_title('Qualifying laps ordered by timestamp')
        axes[0].set_xlabel('Elapsed minutes from session start')
        axes[0].set_ylabel('Lap order')

        for _, row in gap_df.iterrows():
            axes[0].axvspan(row['Gap_Start_Min'], row['Gap_End_Min'], alpha=0.2)
            axes[0].axvline(row['Gap_Mid_Min'], linestyle='--', alpha=0.8)
            axes[0].text(
                row['Gap_Mid_Min'],
                diag['Lap_Order'].max() * 0.95,
                f"{row['Time_Diff']:.1f}m",
                rotation=90,
                va='top',
                ha='center'
            )

        # 2) laps per minute
        axes[1].bar(minute_counts['Minute_Bin'], minute_counts['Lap_Count'], width=0.9)
        axes[1].set_title('Lap count per minute')
        axes[1].set_xlabel('Minute bin')
        axes[1].set_ylabel('Number of laps')

        for _, row in gap_df.iterrows():
            axes[1].axvspan(row['Gap_Start_Min'], row['Gap_End_Min'], alpha=0.2)
            axes[1].axvline(row['Gap_Mid_Min'], linestyle='--', alpha=0.8)

        # 3) gap size between consecutive laps
        axes[2].plot(diag['Lap_Order'], diag['Time_Diff'], marker='o', linewidth=1)
        axes[2].axhline(gap_threshold, linestyle='--')
        axes[2].set_title('Gap size between consecutive laps')
        axes[2].set_xlabel('Lap order')
        axes[2].set_ylabel('Gap in minutes')

        plt.show()

    return diag, gap_df, minute_counts


def pick_quali_boundaries(df_quali, min_gap=6.0):
    quali_time_df = (
        df_quali[['Time']]
        .sort_values('Time')
        .reset_index(drop=True)
        .copy()
    )

    quali_time_df['Time_Minutes_Exact'] = quali_time_df['Time'].dt.total_seconds() / 60
    quali_time_df['Gap_Start_Min'] = quali_time_df['Time_Minutes_Exact'].shift(1)
    quali_time_df['Gap_End_Min'] = quali_time_df['Time_Minutes_Exact']
    quali_time_df['Time_Diff'] = (
        quali_time_df['Gap_End_Min'] - quali_time_df['Gap_Start_Min']
    )

    gap_df = quali_time_df.loc[
        quali_time_df['Gap_Start_Min'].notna() &
        (quali_time_df['Time_Diff'] >= min_gap),
        ['Gap_Start_Min', 'Gap_End_Min', 'Time_Diff']
    ].reset_index(drop=True)

    if len(gap_df) < 2:
        return None, gap_df

    time_min = df_quali['Time'].dt.total_seconds() / 60

    best = None
    best_score = np.inf

    for i in range(len(gap_df)):
        for j in range(i + 1, len(gap_df)):
            q1_end = gap_df.loc[i, 'Gap_Start_Min']
            q2_end = gap_df.loc[j, 'Gap_Start_Min']

            session = np.select(
                [
                    time_min <= q1_end,
                    (time_min > q1_end) & (time_min <= q2_end),
                    time_min > q2_end
                ],
                ['Q1', 'Q2', 'Q3'],
                default='Unknown'
            )

            tmp = df_quali[['Driver']].copy()
            tmp['Session'] = session

            counts = tmp.groupby('Session')['Driver'].nunique().to_dict()
            n_q1 = counts.get('Q1', 0)
            n_q2 = counts.get('Q2', 0)
            n_q3 = counts.get('Q3', 0)

            score = 0

            if not (19 <= n_q1 <= 20):
                score += 20

            if not (13 <= n_q2 <= 15):
                score += 20

            if not (8 <= n_q3 <= 10):
                score += 20

            score += abs(n_q1 - 20)
            score += abs(n_q2 - 15)
            score += abs(n_q3 - 10)

            score -= gap_df.loc[i, 'Time_Diff'] * 0.2
            score -= gap_df.loc[j, 'Time_Diff'] * 0.2

            if score < best_score:
                best_score = score
                best = {
                    'q1_end_min': q1_end,
                    'q2_end_min': q2_end,
                    'gap_1': gap_df.loc[i].to_dict(),
                    'gap_2': gap_df.loc[j].to_dict(),
                    'counts': {'Q1': n_q1, 'Q2': n_q2, 'Q3': n_q3},
                    'score': score
                }

    return best, gap_df


def practice_quali_new(
        season=2018,
        gp='Australia',
        timing_paths=[
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 1.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 2.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 3.csv',
            r'..\data\raw\2018\Australian GP\Qualifying\2018-Australian Grand Prix-Qualifying.csv',
            r'..\data\raw\2018\Australian GP\Sprint-Qualifying\2018-Australian Grand Prix-Sprint Qualifying.csv'
        ],
        weather_paths=[
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 1-weather.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 2-weather.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 3-weather.csv',
            r'..\data\raw\2018\Australian GP\Qualifying\2018-Australian Grand Prix-Qualifying-weather.csv',
            r'..\data\raw\2018\Australian GP\Sprint-Qualifying\2018-Australian Grand Prix-Sprint Qualifying-weather.csv'
        ],
        w_vis = True
):
    #timing_paths = [Path(p) for p in timing_paths]
    #weather_paths = [Path(p) for p in weather_paths]

    # -----------------------------
    # helper for practice sessions
    # -----------------------------
    def build_practice_fastest(df_session, master_driver, suffix):
        grouped = df_session.groupby(
            ['Driver', 'Team', 'GP', 'Rule_Era', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp']
        )['laptime_sum_sectortimes'].min().to_frame().reset_index().sort_values(
            'laptime_sum_sectortimes', ascending=True
        )

        merge_cols = list(grouped.columns)
        loc_cols = list(grouped.columns) + ['Compound']

        grouped['LapTimeDiff'] = (
            grouped['laptime_sum_sectortimes'] - grouped['laptime_sum_sectortimes'].min()
        )

        fastest = pd.merge(grouped, df_session.loc[:, loc_cols], on=merge_cols, how='left')

        fastest_idx = fastest.groupby(['Driver', 'Team'])['laptime_sum_sectortimes'].idxmin()
        fastest = (
            fastest.loc[fastest_idx]
            .sort_values('laptime_sum_sectortimes', ascending=True)
            .reset_index(drop=True)
        )

        fastest = pd.merge(master_driver, fastest, on=['Driver', 'Team', 'GP', 'Rule_Era'], how='left')

        group_col = ['Driver', 'Team', 'GP', 'Rule_Era']
        practice_col = [c for c in fastest.columns if c not in group_col]
        fastest = fastest.rename(columns={c: f'{c}_{suffix}' for c in practice_col})

        fastest = fastest.sort_values(f'laptime_sum_sectortimes_{suffix}', ascending=True).reset_index(drop=True)
        return fastest

    # -----------------------------
    # PRACTICE 1
    # -----------------------------
    df_p1 = None
    if timing_paths[0].exists() and weather_paths[0].exists():
        practice_1 = timing_paths[0]
        practice_1_weather = weather_paths[0]

        df_p1 = pd.read_csv(practice_1).iloc[:, 1:].copy()
        df_p1['Season'] = season
        df_p1['Rule_Era'] = df_p1.apply(f1_rule_era, axis=1)
        df_p1['Abandoned_Lap'] = df_p1.apply(abandoned_lap, axis=1)
        df_p1['GP'] = gp
        df_p1['Time'] = pd.to_timedelta(df_p1['Time'])
        df_p1['Time_Minutes'] = np.ceil(df_p1['Time'].dt.total_seconds() / 60)

        df_p1_weather = pd.read_csv(practice_1_weather).iloc[:, 1:].copy()
        df_p1_weather['Time'] = pd.to_timedelta(df_p1_weather['Time'])
        df_p1_weather['Time_Minutes'] = np.ceil(df_p1_weather['Time'].dt.total_seconds() / 60)
        df_p1_weather['Rainfall'] = df_p1_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_p1_weather.drop(['Time'], axis=1, inplace=True)

        df_p1 = pd.merge(df_p1, df_p1_weather, on='Time_Minutes', how='inner')
        df_p1 = df_p1[df_p1['Abandoned_Lap'].eq(0)].copy()

        if df_p1['laptime_sum_sectortimes'].notna().any():
            slowest_lap_p1 = df_p1['laptime_sum_sectortimes'].idxmax()
            fillna_input = df_p1.loc[slowest_lap_p1, df_p1.columns[2:]]
            df_p1 = df_p1.fillna(fillna_input).copy()

    if df_p1 is None:
        raise ValueError(f"P1 is required but missing for {gp}-{season}")

    # -----------------------------
    # PRACTICE 2
    # -----------------------------
    df_p2 = None
    if timing_paths[1].exists() and weather_paths[1].exists():
        practice_2 = timing_paths[1]
        practice_2_weather = weather_paths[1]

        df_p2 = pd.read_csv(practice_2).iloc[:, 1:].copy()
        df_p2['Season'] = season
        df_p2['Rule_Era'] = df_p2.apply(f1_rule_era, axis=1)
        df_p2['Abandoned_Lap'] = df_p2.apply(abandoned_lap, axis=1)
        df_p2['GP'] = gp
        df_p2['Time'] = pd.to_timedelta(df_p2['Time'])
        df_p2['Time_Minutes'] = np.ceil(df_p2['Time'].dt.total_seconds() / 60)

        df_p2_weather = pd.read_csv(practice_2_weather).iloc[:, 1:].copy()
        df_p2_weather['Time'] = pd.to_timedelta(df_p2_weather['Time'])
        df_p2_weather['Time_Minutes'] = np.ceil(df_p2_weather['Time'].dt.total_seconds() / 60)
        df_p2_weather['Rainfall'] = df_p2_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_p2_weather.drop(['Time'], axis=1, inplace=True)

        df_p2 = pd.merge(df_p2, df_p2_weather, on='Time_Minutes', how='inner')
        df_p2 = df_p2[df_p2['Abandoned_Lap'].eq(0)].copy()

        if df_p2['laptime_sum_sectortimes'].notna().any():
            slowest_lap_p2 = df_p2['laptime_sum_sectortimes'].idxmax()
            fillna_input = df_p2.loc[slowest_lap_p2, df_p2.columns[2:]]
            df_p2 = df_p2.fillna(fillna_input).copy()

    # -----------------------------
    # PRACTICE 3
    # -----------------------------
    df_p3 = None
    if timing_paths[2].exists() and weather_paths[2].exists():
        practice_3 = timing_paths[2]
        practice_3_weather = weather_paths[2]

        df_p3 = pd.read_csv(practice_3).iloc[:, 1:].copy()
        df_p3['Season'] = season
        df_p3['Rule_Era'] = df_p3.apply(f1_rule_era, axis=1)
        df_p3['Abandoned_Lap'] = df_p3.apply(abandoned_lap, axis=1)
        df_p3['GP'] = gp
        df_p3['Time'] = pd.to_timedelta(df_p3['Time'])
        df_p3['Time_Minutes'] = np.ceil(df_p3['Time'].dt.total_seconds() / 60)

        df_p3_weather = pd.read_csv(practice_3_weather).iloc[:, 1:].copy()
        df_p3_weather['Time'] = pd.to_timedelta(df_p3_weather['Time'])
        df_p3_weather['Time_Minutes'] = np.ceil(df_p3_weather['Time'].dt.total_seconds() / 60)
        df_p3_weather['Rainfall'] = df_p3_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_p3_weather.drop(['Time'], axis=1, inplace=True)

        df_p3 = pd.merge(df_p3, df_p3_weather, on='Time_Minutes', how='inner')
        df_p3 = df_p3[df_p3['Abandoned_Lap'].eq(0)].copy()

        if df_p3['laptime_sum_sectortimes'].notna().any():
            slowest_lap_p3 = df_p3['laptime_sum_sectortimes'].idxmax()
            fillna_input = df_p3.loc[slowest_lap_p3, df_p3.columns[2:]]
            df_p3 = df_p3.fillna(fillna_input).copy()

    # -----------------------------
    # MASTER DRIVER TABLE (practice only here)
    # -----------------------------
    practice_frames = [df for df in [df_p1, df_p2, df_p3] if df is not None]

    master_driver = (
        pd.concat(
            [df[['Driver', 'Team', 'GP', 'Rule_Era']] for df in practice_frames],
            ignore_index=True
        )
        .drop_duplicates()
        .copy()
    )

    base_keys = master_driver.copy()

    # -----------------------------
    # placeholders for missing P2 / P3
    # -----------------------------
    df_p2_na = base_keys.copy()
    for col in [
        'AirTemp_p2', 'Humidity_p2', 'Pressure_p2', 'Rainfall_p2',
        'TrackTemp_p2', 'laptime_sum_sectortimes_p2', 'LapTimeDiff_p2', 'Compound_p2'
    ]:
        df_p2_na[col] = np.nan
    df_p2_na = df_p2_na[
        ['Driver', 'Team', 'GP', 'Rule_Era',
         'AirTemp_p2', 'Humidity_p2', 'Pressure_p2', 'Rainfall_p2',
         'TrackTemp_p2', 'laptime_sum_sectortimes_p2', 'LapTimeDiff_p2', 'Compound_p2']
    ].copy()

    df_p3_na = base_keys.copy()
    for col in [
        'AirTemp_p3', 'Humidity_p3', 'Pressure_p3', 'Rainfall_p3',
        'TrackTemp_p3', 'laptime_sum_sectortimes_p3', 'LapTimeDiff_p3', 'Compound_p3'
    ]:
        df_p3_na[col] = np.nan
    df_p3_na = df_p3_na[
        ['Driver', 'Team', 'GP', 'Rule_Era',
         'AirTemp_p3', 'Humidity_p3', 'Pressure_p3', 'Rainfall_p3',
         'TrackTemp_p3', 'laptime_sum_sectortimes_p3', 'LapTimeDiff_p3', 'Compound_p3']
    ].copy()

    # -----------------------------
    # PRACTICE FASTEST TABLES
    # -----------------------------
    p1_fast_time = build_practice_fastest(df_p1, master_driver, 'p1')

    if df_p2 is None:
        p2_fast_time = df_p2_na.copy()
    else:
        p2_fast_time = build_practice_fastest(df_p2, master_driver, 'p2')

    if df_p3 is None:
        p3_fast_time = df_p3_na.copy()
    else:
        p3_fast_time = build_practice_fastest(df_p3, master_driver, 'p3')

    # -----------------------------
    # PRACTICE OVERVIEW
    # -----------------------------
    df_training_overview = p1_fast_time.copy()
    for t in [p2_fast_time, p3_fast_time]:
        df_training_overview = pd.merge(
            df_training_overview,
            t,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )

    # -----------------------------
    # Sprint-Qualifying
    # -----------------------------
    df_sprint_quali = None
    df_sprint_quali_overview = None

    has_sprint_files = (
        len(timing_paths) > 4 and
        len(weather_paths) > 4 and
        timing_paths[4].exists() and
        weather_paths[4].exists()
    )

    if has_sprint_files:
        quali_sprint = timing_paths[4]
        quali_sprint_weather = weather_paths[4]

        df_sprint_quali = pd.read_csv(quali_sprint).iloc[:, 1:].copy()
        df_sprint_quali['Season'] = season
        df_sprint_quali['Rule_Era'] = df_sprint_quali.apply(f1_rule_era, axis=1)
        df_sprint_quali['Abandoned_Lap'] = df_sprint_quali.apply(abandoned_lap, axis=1)
        df_sprint_quali['GP'] = gp
        df_sprint_quali['Time'] = pd.to_timedelta(df_sprint_quali['Time'])
        df_sprint_quali['Time_Minutes'] = np.ceil(df_sprint_quali['Time'].dt.total_seconds() / 60)

        df_sprint_quali_weather = pd.read_csv(quali_sprint_weather).iloc[:, 1:].copy()
        df_sprint_quali_weather['Time'] = pd.to_timedelta(df_sprint_quali_weather['Time'])
        df_sprint_quali_weather['Time_Minutes'] = np.ceil(df_sprint_quali_weather['Time'].dt.total_seconds() / 60)
        df_sprint_quali_weather['Rainfall'] = df_sprint_quali_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_sprint_quali_weather.drop(['Time'], axis=1, inplace=True)

        df_sprint_quali = pd.merge(df_sprint_quali, df_sprint_quali_weather, on='Time_Minutes', how='inner')
        df_sprint_quali = df_sprint_quali[df_sprint_quali['Abandoned_Lap'].eq(0)].copy()

        boundary_result, gap_df = pick_quali_boundaries(df_sprint_quali, min_gap=6.0)

        if boundary_result is not None:
            sq1_end_min = boundary_result['q1_end_min']
            sq2_end_min = boundary_result['q2_end_min']
        else:
            quali_time_df = (
                df_sprint_quali[['Time']]
                .sort_values('Time', ascending=True)
                .reset_index(drop=True)
            )
            max_time = quali_time_df['Time'].dt.total_seconds().max() / 60
            sq1_end_min = min(19.0, max_time)
            sq2_end_min = min(29.0, max_time)

        time_min = df_sprint_quali['Time'].dt.total_seconds() / 60

        df_sprint_quali['Quali_Session'] = np.select(
            [
                time_min <= sq1_end_min,
                (time_min > sq1_end_min) & (time_min <= sq2_end_min),
                time_min > sq2_end_min
            ],
            ['SQ1', 'SQ2', 'SQ3'],
            default='Unknown'
        )

        df_sprint_quali = df_sprint_quali[df_sprint_quali['Quali_Session'] != 'Unknown'].copy()

        sq3_rain_mask = (
            (df_sprint_quali['Quali_Session'] == 'SQ3') &
            (df_sprint_quali['Rainfall'] == 1)
        )

        if sq3_rain_mask.sum() > 10:
            df_sprint_quali = df_sprint_quali[
                (df_sprint_quali['Quali_Session'] != 'SQ3') | sq3_rain_mask
            ].copy()

        sprint_fast = df_sprint_quali.groupby(
            ['Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session',
             'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp']
        )['laptime_sum_sectortimes'].min().reset_index()

        if not sprint_fast.empty:
            sprint_fast['LapTimeDiff'] = (
                sprint_fast.groupby('Quali_Session')['laptime_sum_sectortimes']
                .transform(lambda s: s - s.min())
            )

            merge_cols = [
                'Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session',
                'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
                'laptime_sum_sectortimes'
            ]

            q_fast_time = pd.merge(
                sprint_fast,
                df_sprint_quali.loc[:, merge_cols + ['Compound']],
                on=merge_cols,
                how='left'
            )

            fastest_idx = q_fast_time.groupby(
                ['Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session']
            )['laptime_sum_sectortimes'].idxmin()

            q_fast_time = q_fast_time.loc[fastest_idx].reset_index(drop=True)

            session_order = {'SQ1': 1, 'SQ2': 2, 'SQ3': 3}
            q_fast_time['session_rank'] = q_fast_time['Quali_Session'].map(session_order)

            deepest_idx = q_fast_time.groupby(
                ['Driver', 'Team', 'GP', 'Rule_Era']
            )['session_rank'].idxmax()

            df_sprint_quali_overview = (
                q_fast_time.loc[deepest_idx]
                .drop(columns='session_rank')
                .rename(columns={'Quali_Session': 'Sprint-Session'})
                .sort_values('laptime_sum_sectortimes')
                .reset_index(drop=True)
            )

            group_col = ['Driver', 'Team', 'GP', 'Rule_Era', 'Sprint-Session']
            sprint_col = [c for c in df_sprint_quali_overview.columns if c not in group_col]

            df_sprint_quali_overview = df_sprint_quali_overview.rename(
                columns={qc: f'{qc}_sprint_quali' for qc in sprint_col}
            )

    # -----------------------------
    # QUALIFYING
    # -----------------------------
    df_quali = None
    df_quali_overview = None

    if timing_paths[3].exists() and weather_paths[3].exists():
        quali = timing_paths[3]
        quali_weather = weather_paths[3]

        df_quali = pd.read_csv(quali).iloc[:, 1:].copy()
        df_quali['Season'] = season
        df_quali['Rule_Era'] = df_quali.apply(f1_rule_era, axis=1)
        df_quali['Abandoned_Lap'] = df_quali.apply(abandoned_lap, axis=1)
        df_quali['GP'] = gp
        df_quali['Time'] = pd.to_timedelta(df_quali['Time'])
        df_quali['Time_Minutes'] = np.ceil(df_quali['Time'].dt.total_seconds() / 60)
        df_quali = df_quali[df_quali['IsAccurate'] == True].copy()

        df_quali_weather = pd.read_csv(quali_weather).iloc[:, 1:].copy()
        df_quali_weather['Time'] = pd.to_timedelta(df_quali_weather['Time'])
        df_quali_weather['Time_Minutes'] = np.ceil(df_quali_weather['Time'].dt.total_seconds() / 60)
        df_quali_weather['Rainfall'] = df_quali_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_quali_weather.drop(['Time'], axis=1, inplace=True)

        df_quali = pd.merge(df_quali, df_quali_weather, on='Time_Minutes', how='inner')
        df_quali = df_quali[df_quali['Abandoned_Lap'].eq(0)].copy()

        quali_diag, quali_gaps, quali_minute_counts = inspect_quali_time_distribution(
            df_quali,
            gp=gp,
            season=season,
            gap_threshold=6.0,
            top_n_gaps=20,
            w_vis=w_vis
        )

        boundary_result, gap_df = pick_quali_boundaries(df_quali, min_gap=6.0)

        if boundary_result is not None:
            q1_end_min = boundary_result['q1_end_min']
            q2_end_min = boundary_result['q2_end_min']
        else:
            quali_time_df = (
                df_quali[['Time']]
                .sort_values('Time', ascending=True)
                .reset_index(drop=True)
            )
            max_time = quali_time_df['Time'].dt.total_seconds().max() / 60
            q1_end_min = min(25.0, max_time)
            q2_end_min = min(40.0, max_time)

        time_min = df_quali['Time'].dt.total_seconds() / 60

        df_quali['Quali_Session'] = np.select(
            [
                time_min <= q1_end_min,
                (time_min > q1_end_min) & (time_min <= q2_end_min),
                time_min > q2_end_min
            ],
            ['Q1', 'Q2', 'Q3'],
            default='Unknown'
        )

        df_quali = df_quali[df_quali['Quali_Session'] != 'Unknown'].copy()

        q3_rain_mask = (
            (df_quali['Quali_Session'] == 'Q3') &
            (df_quali['Rainfall'] == 1)
        )

        if q3_rain_mask.sum() > 10:
            df_quali = df_quali[
                (df_quali['Quali_Session'] != 'Q3') | q3_rain_mask
            ].copy()

        quali_fast = df_quali.groupby(
            ['Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session',
             'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp']
        )['laptime_sum_sectortimes'].min().reset_index()

        if not quali_fast.empty:
            quali_fast['LapTimeDiff'] = (
                quali_fast.groupby('Quali_Session')['laptime_sum_sectortimes']
                .transform(lambda s: s - s.min())
            )

            merge_cols = [
                'Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session',
                'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
                'laptime_sum_sectortimes'
            ]

            q_fast_time = pd.merge(
                quali_fast,
                df_quali.loc[:, merge_cols + ['Compound']],
                on=merge_cols,
                how='left'
            )

            fastest_idx = q_fast_time.groupby(
                ['Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session']
            )['laptime_sum_sectortimes'].idxmin()

            q_fast_time = q_fast_time.loc[fastest_idx].reset_index(drop=True)

            session_order = {'Q1': 1, 'Q2': 2, 'Q3': 3}
            q_fast_time['session_rank'] = q_fast_time['Quali_Session'].map(session_order)

            deepest_idx = q_fast_time.groupby(
                ['Driver', 'Team', 'GP', 'Rule_Era']
            )['session_rank'].idxmax()

            df_quali_overview = (
                q_fast_time.loc[deepest_idx]
                .drop(columns='session_rank')
                .rename(columns={'Quali_Session': 'Session'})
                .sort_values('laptime_sum_sectortimes')
                .reset_index(drop=True)
            )

            group_col = ['Driver', 'Team', 'GP', 'Rule_Era', 'Session']
            quali_col = [c for c in df_quali_overview.columns if c not in group_col]

            df_quali_overview = df_quali_overview.rename(
                columns={qc: f'{qc}_quali' for qc in quali_col}
            )

            df_quali_overview = df_quali_overview.sort_values(
                'laptime_sum_sectortimes_quali'
            ).reset_index(drop=True)

    # -----------------------------
    # FINAL KEY TABLE: union of all available overview tables
    # -----------------------------
    all_key_frames = [
        df_training_overview[['Driver', 'Team', 'GP', 'Rule_Era']]
    ]

    if df_sprint_quali_overview is not None:
        all_key_frames.append(
            df_sprint_quali_overview[['Driver', 'Team', 'GP', 'Rule_Era']]
        )

    if df_quali_overview is not None:
        all_key_frames.append(
            df_quali_overview[['Driver', 'Team', 'GP', 'Rule_Era']]
        )

    all_keys_final = (
        pd.concat(all_key_frames, ignore_index=True)
        .drop_duplicates()
        .copy()
    )

    # -----------------------------
    # sprint placeholder on final key table
    # -----------------------------
    df_sprint_quali_na = all_keys_final.copy()
    df_sprint_quali_na['Sprint-Session'] = np.nan
    for col in [
        'AirTemp_sprint_quali', 'Humidity_sprint_quali', 'Pressure_sprint_quali', 'Rainfall_sprint_quali',
        'TrackTemp_sprint_quali', 'laptime_sum_sectortimes_sprint_quali', 'LapTimeDiff_sprint_quali', 'Compound_sprint_quali'
    ]:
        df_sprint_quali_na[col] = np.nan

    df_sprint_quali_na = df_sprint_quali_na[
        ['Driver', 'Team', 'GP', 'Rule_Era', 'Sprint-Session',
         'AirTemp_sprint_quali', 'Humidity_sprint_quali', 'Pressure_sprint_quali', 'Rainfall_sprint_quali',
         'TrackTemp_sprint_quali', 'laptime_sum_sectortimes_sprint_quali', 'LapTimeDiff_sprint_quali', 'Compound_sprint_quali']
    ].copy()

    # -----------------------------
    # qualifying placeholder on final key table
    # -----------------------------
    df_quali_overview_na = all_keys_final.copy()
    df_quali_overview_na['Session'] = np.nan
    for col in [
        'AirTemp_quali', 'Humidity_quali', 'Pressure_quali', 'Rainfall_quali',
        'TrackTemp_quali', 'laptime_sum_sectortimes_quali', 'LapTimeDiff_quali', 'Compound_quali'
    ]:
        df_quali_overview_na[col] = np.nan

    df_quali_overview_na = df_quali_overview_na[
        ['Driver', 'Team', 'GP', 'Rule_Era', 'Session',
         'AirTemp_quali', 'Humidity_quali', 'Pressure_quali', 'Rainfall_quali',
         'TrackTemp_quali', 'laptime_sum_sectortimes_quali', 'LapTimeDiff_quali', 'Compound_quali']
    ].copy()

    # -----------------------------
    # FINAL MERGE
    # -----------------------------
    df_final_gp = pd.merge(
        all_keys_final,
        df_training_overview,
        on=['Driver', 'Team', 'GP', 'Rule_Era'],
        how='left'
    )

    if df_sprint_quali_overview is not None:
        df_final_gp = pd.merge(
            df_final_gp,
            df_sprint_quali_overview,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )
    else:
        df_final_gp = pd.merge(
            df_final_gp,
            df_sprint_quali_na,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )

    if df_quali_overview is not None:
        df_final_gp = pd.merge(
            df_final_gp,
            df_quali_overview,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )
    else:
        df_final_gp = pd.merge(
            df_final_gp,
            df_quali_overview_na,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )

    df_final_gp['Season'] = season

    # -----------------------------
    # ERA FLAGS
    # -----------------------------
    df_final_gp['Sprint_Race_Era'] = df_final_gp['Season'].apply(
        lambda x: '2021-2022' if 2021 <= x <= 2022
        else ('2024' if x == 2024 else '2023-2026')
    )

    df_final_gp['Sprint_Weekend'] = df_final_gp['Sprint-Session'].apply(
        lambda x: 0 if pd.isna(x) else 1
    )

    return df_final_gp

def get_sprint_session_name(season: int):
    # adapt this to your real file naming
    if season < 2025:
        return "Sprint Shootout"
    else:
        return "Sprint Qualifying"
    
def build_session_paths(base: Path, season: int, folder_loop_value: str):
    race_name_for_file = folder_loop_value.replace(" GP", "")

    prac_1 = base / "Practice" / f"{season}-{race_name_for_file} Grand Prix-Practice 1.csv"
    prac_2 = base / "Practice" / f"{season}-{race_name_for_file} Grand Prix-Practice 2.csv"
    prac_3 = base / "Practice" / f"{season}-{race_name_for_file} Grand Prix-Practice 3.csv"
    quali = base / "Qualifying" / f"{season}-{race_name_for_file} Grand Prix-Qualifying.csv"

    prac_1_weather = base / "Practice" / f"{season}-{race_name_for_file} Grand Prix-Practice 1-weather.csv"
    prac_2_weather = base / "Practice" / f"{season}-{race_name_for_file} Grand Prix-Practice 2-weather.csv"
    prac_3_weather = base / "Practice" / f"{season}-{race_name_for_file} Grand Prix-Practice 3-weather.csv"
    quali_weather = base / "Qualifying" / f"{season}-{race_name_for_file} Grand Prix-Qualifying-weather.csv"

    timing_paths = [prac_1, prac_2, prac_3, quali]
    weather_paths = [prac_1_weather, prac_2_weather, prac_3_weather, quali_weather]

    sprint_session_name = get_sprint_session_name(season)

    #using get_sprint_session_name() to get the session name, and check if it exists or not
    if sprint_session_name is not None:
        sprint_quali = (
            base
            / "Sprint-Qualifying"
            / f"{season}-{race_name_for_file} Grand Prix-{sprint_session_name}.csv"
        )
        sprint_quali_weather = (
            base
            / "Sprint-Qualifying"
            / f"{season}-{race_name_for_file} Grand Prix-{sprint_session_name}-weather.csv"
        )

        #checking if the variable exists or not
        if sprint_quali.exists() and sprint_quali_weather.exists():
            timing_paths.append(sprint_quali)
            weather_paths.append(sprint_quali_weather)

    return timing_paths, weather_paths

def games_howell_simple_effects(
        data,
        dv,
        compare_factor,
        within_factor,
        min_n = 2,
        correction="holm"
):
    results = []
    for level,group in data.groupby(within_factor):
        counts = group[compare_factor].value_counts()
        valid_levels = counts[counts >= min_n].index
        group = group[group[compare_factor].isin(valid_levels)]

        if group[compare_factor].nunique() < 2:
            continue

        result = pg.pairwise_gameshowell(
            data = group,
            dv=dv,
            between=compare_factor
        )

        result[within_factor] = level
        results.append(result)

        if len(results) == 0:
            return pd.DataFrame()

        out = pd.concat(results,ignore_index=True)

        out['p_corrected_global'] = multipletests(out['pval'],method=correction)[1]

        out['significance'] = np.where(
            out['pval'] < 0.05,
            'Sig',
            'N-Sig'
        )

    return out

def robust_z_score(series):
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = 1e-6
    return 0.6745 *(series - median) / mad

################################## Pipeline ##################################
@dataclass
class Add_Column(BaseEstimator,TransformerMixin):
    col: Sequence[str]
    name_: Sequence[str]

    def fit(self, X:pd.DataFrame, y=None):
        missing_col = [c for c in self.col if c not in X.columns]
        if missing_col:
            raise ValueError(f'Column(s) is/are not in X: {missing_col}')
        return self
    
    def transform(self,X:pd.DataFrame,):
        X = X.copy()
        for c in self.col:
            for n in self.name_:
                X[f'{c}{n}'] = X[c]
        return X


@dataclass
class permutation_rf_feature_selector(
    BaseEstimator,
    TransformerMixin,
):
    task: Literal[
        "regression",
        "classification",
    ] = "regression"

    #column that defines chronological GP groups
    group_column: str = "gp_id"

    #percentage of the most recent training groups used
    #to calculate permutation importance
    validation_fraction: float = 0.20

    #optional fixed number of validation groups.
    #when provided, this overrides validation_fraction.
    n_validation_groups: Optional[int] = None

    n_estimators: int = 200
    n_repeats: int = 5

    random_state: int = 101
    n_jobs: int = -1

    threshold: float = 0.0
    max_features_to_select: Optional[int] = None

    def fit(self, X, y):
        # Because the selector needs gp_id by name,
        # X must be a DataFrame.
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame because "
                f"{self.group_column!r} is required."
            )

        if self.group_column not in X.columns:
            raise ValueError(
                f"{self.group_column!r} was not found in X. "
                "Do not drop it before the feature selector."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1."
            )

        if self.n_validation_groups is not None:
            if self.n_validation_groups < 1:
                raise ValueError(
                    "n_validation_groups must be at least 1."
                )

        X_df = X.copy()
        y_array = np.asarray(y).ravel()

        # Store all input column names
        self.feature_names_in_ = np.asarray(
            X_df.columns,
            dtype=object,
        )

        # gp_id is used only for splitting.
        # It is not a candidate predictive feature.
        self.candidate_features_ = [
            column
            for column in X_df.columns
            if column != self.group_column
        ]

        if len(self.candidate_features_) == 0:
            raise ValueError(
                "No candidate features remain after excluding "
                f"{self.group_column!r}."
            )

        groups = X_df[self.group_column].to_numpy()

        # This works because your gp_id is already
        # numeric and chronologically ordered.
        ordered_groups = np.sort(
            np.unique(groups)
        )

        n_groups = len(ordered_groups)

        if n_groups < 2:
            raise ValueError(
                "At least two distinct groups are required."
            )

        #determine how many complete recent GP groups
        #should be used for permutation importance.
        if self.n_validation_groups is not None:
            n_val_groups = self.n_validation_groups
        else:
            n_val_groups = max(
                1,
                int(np.ceil(
                    self.validation_fraction * n_groups
                )),
            )

        if n_val_groups >= n_groups:
            raise ValueError(
                "The internal validation set would use all groups. "
                "Reduce validation_fraction or "
                "n_validation_groups."
            )

        #older GP groups are used to fit the RF.
        fit_groups = ordered_groups[:-n_val_groups]

        #most recent GP groups are used to calculate
        #permutation importance
        permutation_groups = ordered_groups[-n_val_groups:]

        fit_mask = np.isin(
            groups,
            fit_groups,
        )

        permutation_mask = np.isin(
            groups,
            permutation_groups,
        )

        X_fit = X_df.loc[
            fit_mask,
            self.candidate_features_,
        ]

        y_fit = y_array[fit_mask]

        X_permutation = X_df.loc[
            permutation_mask,
            self.candidate_features_,
        ]

        y_permutation = y_array[permutation_mask]

        if len(X_fit) == 0 or len(X_permutation) == 0:
            raise ValueError(
                "The internal feature-selection split produced "
                "an empty training or validation set."
            )

        if self.task == "regression":
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

            scoring = "neg_mean_absolute_error"

        else:
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

            scoring = "f1_macro"

        # Fit the Random Forest only on the older groups
        self.model_.fit(
            X_fit,
            y_fit,
        )

        # Evaluate permutation importance on separate,
        # more recent GP groups
        result = permutation_importance(
            estimator=self.model_,
            X=X_permutation,
            y=y_permutation,
            scoring=scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        importances = result.importances_mean

        self.importances_ = (
            pd.DataFrame(
                {
                    "feature": self.candidate_features_,
                    "importance": importances,
                }
            )
            .sort_values(
                by="importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        # Initial threshold-based selection
        selected_mask = (
            importances > self.threshold
        )

        # max_features_to_select acts as an upper limit.
        # Only the strongest N features may remain.
        if self.max_features_to_select is not None:
            if self.max_features_to_select < 1:
                raise ValueError(
                    "max_features_to_select must be at least 1."
                )

            n_features = min(
                self.max_features_to_select,
                len(importances),
            )

            top_indices = np.argsort(
                importances
            )[::-1][:n_features]

            top_mask = np.zeros(
                len(importances),
                dtype=bool,
            )

            top_mask[top_indices] = True

            # Feature must pass the threshold and be
            # among the strongest N features.
            selected_mask = (
                selected_mask & top_mask
            )

        # Prevent the selector from returning zero features
        if not selected_mask.any():
            strongest_feature_idx = int(
                np.argmax(importances)
            )

            selected_mask[
                strongest_feature_idx
            ] = True

        self.support_ = selected_mask

        candidate_features_array = np.asarray(
            self.candidate_features_,
            dtype=object,
        )

        self.selected_features_ = (
            candidate_features_array[
                selected_mask
            ]
            .tolist()
        )

        # Store the internal group split for inspection
        self.fit_groups_ = fit_groups
        self.permutation_groups_ = permutation_groups

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            attributes=[
                "support_",
                "selected_features_",
            ],
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        missing_features = set(
            self.selected_features_
        ) - set(X.columns)

        if missing_features:
            raise ValueError(
                "The following selected features are missing "
                f"from X: {sorted(missing_features)}"
            )

        # gp_id is automatically excluded because it is not
        # included in selected_features_.
        return X.loc[
            :,
            self.selected_features_,
        ].copy()

    def get_support(self):
        check_is_fitted(
            self,
            attributes=["support_"],
        )

        return self.support_.copy()

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        check_is_fitted(
            self,
            attributes=["selected_features_"],
        )

        return np.asarray(
            self.selected_features_,
            dtype=object,
        )


@dataclass
class RFPermutationRegressorSelector(
    permutation_rf_feature_selector
):
    task: Literal["regression"] = "regression"


@dataclass
class RFPermutationClassifierSelector(
    permutation_rf_feature_selector
):
    task: Literal["classification"] = "classification"

#How to use it
#selector_reg = RFPermutationRegressorSelector(threshold=0.01)
#selector_reg.fit(X_train, y_train)
#X_train_selected = selector_reg.transform(X_train)
#X_test_selected = selector_reg.transform

@dataclass
class GroupTimeSplit(BaseCrossValidator):
    group_column: str = 'gp_id'
    n_splits: int = 5
    test_group_size: int = 3
    min_train_group_size: int | None = None

    def split(self, X, y=None, groups=None):
        if groups is None:
            if self.group_column not in X.columns:
                raise ValueError(f"{self.group_column} not found in X.")
            groups = X[self.group_column].values

        groups = np.asarray(groups)
        unique_groups = np.sort(np.unique(groups))
        n_groups = len(unique_groups)

        min_train = (
            n_groups - self.n_splits * self.test_group_size
            if self.min_train_group_size is None
            else self.min_train_group_size
        )

        if min_train <= 0:
            raise ValueError("Not enough groups for the requested number of splits.")

        for split_idx in range(self.n_splits):
            train_end = min_train + split_idx * self.test_group_size

            train_groups = unique_groups[:train_end]
            test_groups = unique_groups[train_end: train_end + self.test_group_size]

            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

#creating a def function, which allows us to tune refined parameter ranges
##it respects the numeric type
#def previous_stage_param_range(param,range_per,param_limit=None):
    #if isinstance(param,float) and isinstance(param_limit,list):
        #lower_bound = param * (1-range_per)
        #upper_bound = param * (1+range_per)

        #lower_bound = max(lower_bound,param_limit[0])
        #upper_bound = min(upper_bound,param_limit[1])
        #return lower_bound,upper_bound
    #elif isinstance(param,float):
        #lower_bound = param * (1-range_per)
        #upper_bound = param * (1+range_per)
        #return lower_bound,upper_bound
    #elif isinstance(param,int):
        #lower_bound = int(np.ceil(param * (1-range_per)))
        #upper_bound = int(np.ceil(param * (1+range_per)))
        #return lower_bound,upper_bound
    
def previous_stage_param_range(param, range_per, param_limit=None):
    if param_limit is not None:
        param = np.clip(param, param_limit[0], param_limit[1])

    lower_bound = param * (1 - range_per)
    upper_bound = param * (1 + range_per)

    if param_limit is not None:
        lower_bound = max(lower_bound, param_limit[0])
        upper_bound = min(upper_bound, param_limit[1])

    if isinstance(param, (int, np.integer)):
        lower_bound = int(np.ceil(lower_bound))
        upper_bound = int(np.ceil(upper_bound))

    return lower_bound, upper_bound

#customized learning curve, which respects our gp_id order logic
def group_time_learning_curve(
    estimator,
    X,
    y,
    cv,
    train_fractions=np.linspace(0.1, 1.0, 10),
):
    #checking if indexies of X and y align
    if not X.index.equals(y.index):
        raise ValueError("X and y indices are not aligned.")

    #extract the GP ID for every row
    groups = X[cv.group_column].to_numpy()

    #determine the result-array sizes
    ##n_sizes = 10
    ##n_folds = 5
    ##10 training sizes × 5 folds = 50 models
    n_sizes = len(train_fractions)
    n_folds = cv.get_n_splits()

    #create empty result arrays
    ##with ten sizes and five folds, each array has shape => (10, 5)
    ##the folds already respect chronology
    ###full_train_idx contains the row positions for the full training fold
    train_scores = np.empty((n_sizes, n_folds))
    val_scores = np.empty((n_sizes, n_folds))
    group_sizes = np.empty((n_sizes, n_folds), dtype=int)

    #outer loop=> iterate through CV folds
    ##val_idx contains the row positions for the validation fold
    for fold_idx, (full_train_idx, val_idx) in enumerate(
        cv.split(X, y)
    ):
        # Chronological GP groups available in this training fold
        full_train_groups = np.sort(
            np.unique(groups[full_train_idx])
        )

        for size_idx, fraction in enumerate(train_fractions):
            #calculate the number of complete GPs
            ##suppose there are 149 training GPs => 10%
            ##0.10 * 149 = 14.9
            ##np.ceil() => 14.9 => 15
            n_train_groups = max(
                1,
                int(np.ceil(
                    fraction * len(full_train_groups)
                )),
            )

            #create a row mask for the selected groups
            ##use the most recent complete training groups
            ###this keeps the training window close to validation
            ###so we create a dataframe with the selected group values
            ####in this case, gp_id
            selected_groups = full_train_groups[-n_train_groups:]

            selected_mask = np.isin(
                groups[full_train_idx],
                selected_groups,
            )

            #this returns the row positions belonging to the selected complete GPs
            ##therefore, if one GP has 20 drivers, all 20 rows are included
            train_idx = full_train_idx[selected_mask]

            #clone() creates a new unfitted copy of the pipeline with the same parameters
            #it prevents fitted state from one fold or training size carrying over into the next model
            model = clone(estimator)

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            train_pred = model.predict(
                X.iloc[train_idx]
            )
            val_pred = model.predict(
                X.iloc[val_idx]
            )

            train_scores[size_idx, fold_idx] = (
                mean_absolute_error(
                    y.iloc[train_idx],
                    train_pred,
                )
            )

            val_scores[size_idx, fold_idx] = (
                mean_absolute_error(
                    y.iloc[val_idx],
                    val_pred,
                )
            )

            group_sizes[size_idx, fold_idx] = (
                n_train_groups
            )

    #training folds contain slightly different numbers of groups,
    #so use the average group count for the x-axis.
    mean_group_sizes = group_sizes.mean(axis=1)

    return mean_group_sizes, train_scores, val_scores

class CastColumnsToObject(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = X[col].astype("object")
        return X

#------------------------------------------------ Off-Set -----------------------------------------#
def def_apply_offset_mean(
        df,
        pred='pred',
        offset='offset',
        shrinking=1.0,
        limit=None
):
    result = df.copy()
    correction = result[offset].fillna(0.0)*shrinking

    #here we apply a limit, the limit will be applied when the value is not None
    if limit is not None:
        correction = np.clip(
            correction,
            lower = -limit,
            upper = limit
        )
    result['applied_offset'] = correction
    result['pred_corrected'] = result[pred] + result['applied_offset']
    return result

#------------------------------------------------ Prediction -----------------------------------------#
#creating a CSV for predictions
def practice_quali_new_pred(
        season=2018,
        gp='Australia',
        timing_paths=[
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 1.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 2.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 3.csv',
            r'..\data\raw\2018\Australian GP\Sprint-Qualifying\2018-Australian Grand Prix-Sprint Qualifying.csv'
        ],
        weather_paths=[
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 1-weather.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 2-weather.csv',
            r'..\data\raw\2018\Australian GP\Practice\2018-Australian Grand Prix-Practice 3-weather.csv',
            r'..\data\raw\2018\Australian GP\Sprint-Qualifying\2018-Australian Grand Prix-Sprint Qualifying-weather.csv'
        ],
        w_vis = True
):
    timing_paths = [Path(p) for p in timing_paths]
    weather_paths = [Path(p) for p in weather_paths]

    # -----------------------------
    # helper for practice sessions
    # -----------------------------
    def build_practice_fastest_pred(df_session, master_driver, suffix):
        grouped = df_session.groupby(
            ['Driver', 'Team', 'GP', 'Rule_Era', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp']
        )['laptime_sum_sectortimes'].min().to_frame().reset_index().sort_values(
            'laptime_sum_sectortimes', ascending=True
        )

        merge_cols = list(grouped.columns)
        loc_cols = list(grouped.columns) + ['Compound']

        grouped['LapTimeDiff'] = (
            grouped['laptime_sum_sectortimes'] - grouped['laptime_sum_sectortimes'].min()
        )

        fastest = pd.merge(grouped, df_session.loc[:, loc_cols], on=merge_cols, how='left')

        fastest_idx = fastest.groupby(['Driver', 'Team'])['laptime_sum_sectortimes'].idxmin()
        fastest = (
            fastest.loc[fastest_idx]
            .sort_values('laptime_sum_sectortimes', ascending=True)
            .reset_index(drop=True)
        )

        fastest = pd.merge(master_driver, fastest, on=['Driver', 'Team', 'GP', 'Rule_Era'], how='left')

        group_col = ['Driver', 'Team', 'GP', 'Rule_Era']
        practice_col = [c for c in fastest.columns if c not in group_col]
        fastest = fastest.rename(columns={c: f'{c}_{suffix}' for c in practice_col})

        fastest = fastest.sort_values(f'laptime_sum_sectortimes_{suffix}', ascending=True).reset_index(drop=True)
        return fastest

    # -----------------------------
    # PRACTICE 1
    # -----------------------------
    df_p1 = None
    if timing_paths[0].exists() and weather_paths[0].exists():
        practice_1 = timing_paths[0]
        practice_1_weather = weather_paths[0]

        df_p1 = pd.read_csv(practice_1).iloc[:, 1:].copy()
        df_p1['Season'] = season
        df_p1['Rule_Era'] = df_p1.apply(f1_rule_era, axis=1)
        df_p1['Abandoned_Lap'] = df_p1.apply(abandoned_lap, axis=1)
        df_p1['GP'] = gp
        df_p1['Time'] = pd.to_timedelta(df_p1['Time'])
        df_p1['Time_Minutes'] = np.ceil(df_p1['Time'].dt.total_seconds() / 60)

        df_p1_weather = pd.read_csv(practice_1_weather).iloc[:, 1:].copy()
        df_p1_weather['Time'] = pd.to_timedelta(df_p1_weather['Time'])
        df_p1_weather['Time_Minutes'] = np.ceil(df_p1_weather['Time'].dt.total_seconds() / 60)
        df_p1_weather['Rainfall'] = df_p1_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_p1_weather.drop(['Time'], axis=1, inplace=True)

        df_p1 = pd.merge(df_p1, df_p1_weather, on='Time_Minutes', how='inner')
        df_p1 = df_p1[df_p1['Abandoned_Lap'].eq(0)].copy()

        if df_p1['laptime_sum_sectortimes'].notna().any():
            slowest_lap_p1 = df_p1['laptime_sum_sectortimes'].idxmax()
            fillna_input = df_p1.loc[slowest_lap_p1, df_p1.columns[2:]]
            df_p1 = df_p1.fillna(fillna_input).copy()

    if df_p1 is None:
        raise ValueError(f"P1 is required but missing for {gp}-{season}")

    # -----------------------------
    # PRACTICE 2
    # -----------------------------
    df_p2 = None
    if timing_paths[1].exists() and weather_paths[1].exists():
        practice_2 = timing_paths[1]
        practice_2_weather = weather_paths[1]

        df_p2 = pd.read_csv(practice_2).iloc[:, 1:].copy()
        df_p2['Season'] = season
        df_p2['Rule_Era'] = df_p2.apply(f1_rule_era, axis=1)
        df_p2['Abandoned_Lap'] = df_p2.apply(abandoned_lap, axis=1)
        df_p2['GP'] = gp
        df_p2['Time'] = pd.to_timedelta(df_p2['Time'])
        df_p2['Time_Minutes'] = np.ceil(df_p2['Time'].dt.total_seconds() / 60)

        df_p2_weather = pd.read_csv(practice_2_weather).iloc[:, 1:].copy()
        df_p2_weather['Time'] = pd.to_timedelta(df_p2_weather['Time'])
        df_p2_weather['Time_Minutes'] = np.ceil(df_p2_weather['Time'].dt.total_seconds() / 60)
        df_p2_weather['Rainfall'] = df_p2_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_p2_weather.drop(['Time'], axis=1, inplace=True)

        df_p2 = pd.merge(df_p2, df_p2_weather, on='Time_Minutes', how='inner')
        df_p2 = df_p2[df_p2['Abandoned_Lap'].eq(0)].copy()

        if df_p2['laptime_sum_sectortimes'].notna().any():
            slowest_lap_p2 = df_p2['laptime_sum_sectortimes'].idxmax()
            fillna_input = df_p2.loc[slowest_lap_p2, df_p2.columns[2:]]
            df_p2 = df_p2.fillna(fillna_input).copy()

    # -----------------------------
    # PRACTICE 3
    # -----------------------------
    df_p3 = None
    if timing_paths[2].exists() and weather_paths[2].exists():
        practice_3 = timing_paths[2]
        practice_3_weather = weather_paths[2]

        df_p3 = pd.read_csv(practice_3).iloc[:, 1:].copy()
        df_p3['Season'] = season
        df_p3['Rule_Era'] = df_p3.apply(f1_rule_era, axis=1)
        df_p3['Abandoned_Lap'] = df_p3.apply(abandoned_lap, axis=1)
        df_p3['GP'] = gp
        df_p3['Time'] = pd.to_timedelta(df_p3['Time'])
        df_p3['Time_Minutes'] = np.ceil(df_p3['Time'].dt.total_seconds() / 60)

        df_p3_weather = pd.read_csv(practice_3_weather).iloc[:, 1:].copy()
        df_p3_weather['Time'] = pd.to_timedelta(df_p3_weather['Time'])
        df_p3_weather['Time_Minutes'] = np.ceil(df_p3_weather['Time'].dt.total_seconds() / 60)
        df_p3_weather['Rainfall'] = df_p3_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_p3_weather.drop(['Time'], axis=1, inplace=True)

        df_p3 = pd.merge(df_p3, df_p3_weather, on='Time_Minutes', how='inner')
        df_p3 = df_p3[df_p3['Abandoned_Lap'].eq(0)].copy()

        if df_p3['laptime_sum_sectortimes'].notna().any():
            slowest_lap_p3 = df_p3['laptime_sum_sectortimes'].idxmax()
            fillna_input = df_p3.loc[slowest_lap_p3, df_p3.columns[2:]]
            df_p3 = df_p3.fillna(fillna_input).copy()

    # -----------------------------
    # MASTER DRIVER TABLE (practice only here)
    # -----------------------------
    practice_frames = [df for df in [df_p1, df_p2, df_p3] if df is not None]

    master_driver = (
        pd.concat(
            [df[['Driver', 'Team', 'GP', 'Rule_Era']] for df in practice_frames],
            ignore_index=True
        )
        .drop_duplicates()
        .copy()
    )

    base_keys = master_driver.copy()

    # -----------------------------
    # placeholders for missing P2 / P3
    # -----------------------------
    df_p2_na = base_keys.copy()
    for col in [
        'AirTemp_p2', 'Humidity_p2', 'Pressure_p2', 'Rainfall_p2',
        'TrackTemp_p2', 'laptime_sum_sectortimes_p2', 'LapTimeDiff_p2', 'Compound_p2'
    ]:
        df_p2_na[col] = np.nan
    df_p2_na = df_p2_na[
        ['Driver', 'Team', 'GP', 'Rule_Era',
         'AirTemp_p2', 'Humidity_p2', 'Pressure_p2', 'Rainfall_p2',
         'TrackTemp_p2', 'laptime_sum_sectortimes_p2', 'LapTimeDiff_p2', 'Compound_p2']
    ].copy()

    df_p3_na = base_keys.copy()
    for col in [
        'AirTemp_p3', 'Humidity_p3', 'Pressure_p3', 'Rainfall_p3',
        'TrackTemp_p3', 'laptime_sum_sectortimes_p3', 'LapTimeDiff_p3', 'Compound_p3'
    ]:
        df_p3_na[col] = np.nan
    df_p3_na = df_p3_na[
        ['Driver', 'Team', 'GP', 'Rule_Era',
         'AirTemp_p3', 'Humidity_p3', 'Pressure_p3', 'Rainfall_p3',
         'TrackTemp_p3', 'laptime_sum_sectortimes_p3', 'LapTimeDiff_p3', 'Compound_p3']
    ].copy()

    # -----------------------------
    # PRACTICE FASTEST TABLES
    # -----------------------------
    p1_fast_time = build_practice_fastest_pred(df_p1, master_driver, 'p1')

    if df_p2 is None:
        p2_fast_time = df_p2_na.copy()
    else:
        p2_fast_time = build_practice_fastest_pred(df_p2, master_driver, 'p2')

    if df_p3 is None:
        p3_fast_time = df_p3_na.copy()
    else:
        p3_fast_time = build_practice_fastest_pred(df_p3, master_driver, 'p3')

    # -----------------------------
    # PRACTICE OVERVIEW
    # -----------------------------
    df_training_overview = p1_fast_time.copy()
    for t in [p2_fast_time, p3_fast_time]:
        df_training_overview = pd.merge(
            df_training_overview,
            t,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )

    # -----------------------------
    # Sprint-Qualifying
    # -----------------------------
    df_sprint_quali = None
    df_sprint_quali_overview = None

    has_sprint_files = (
        len(timing_paths) > 4 and
        len(weather_paths) > 4 and
        timing_paths[4].exists() and
        weather_paths[4].exists()
    )

    if has_sprint_files:
        quali_sprint = timing_paths[4]
        quali_sprint_weather = weather_paths[4]

        df_sprint_quali = pd.read_csv(quali_sprint).iloc[:, 1:].copy()
        df_sprint_quali['Season'] = season
        df_sprint_quali['Rule_Era'] = df_sprint_quali.apply(f1_rule_era, axis=1)
        df_sprint_quali['Abandoned_Lap'] = df_sprint_quali.apply(abandoned_lap, axis=1)
        df_sprint_quali['GP'] = gp
        df_sprint_quali['Time'] = pd.to_timedelta(df_sprint_quali['Time'])
        df_sprint_quali['Time_Minutes'] = np.ceil(df_sprint_quali['Time'].dt.total_seconds() / 60)

        df_sprint_quali_weather = pd.read_csv(quali_sprint_weather).iloc[:, 1:].copy()
        df_sprint_quali_weather['Time'] = pd.to_timedelta(df_sprint_quali_weather['Time'])
        df_sprint_quali_weather['Time_Minutes'] = np.ceil(df_sprint_quali_weather['Time'].dt.total_seconds() / 60)
        df_sprint_quali_weather['Rainfall'] = df_sprint_quali_weather['Rainfall'].apply(lambda x: 1 if x else 0)
        df_sprint_quali_weather.drop(['Time'], axis=1, inplace=True)

        df_sprint_quali = pd.merge(df_sprint_quali, df_sprint_quali_weather, on='Time_Minutes', how='inner')
        df_sprint_quali = df_sprint_quali[df_sprint_quali['Abandoned_Lap'].eq(0)].copy()

        boundary_result, gap_df = pick_quali_boundaries(df_sprint_quali, min_gap=6.0)

        if boundary_result is not None:
            sq1_end_min = boundary_result['q1_end_min']
            sq2_end_min = boundary_result['q2_end_min']
        else:
            quali_time_df = (
                df_sprint_quali[['Time']]
                .sort_values('Time', ascending=True)
                .reset_index(drop=True)
            )
            max_time = quali_time_df['Time'].dt.total_seconds().max() / 60
            sq1_end_min = min(19.0, max_time)
            sq2_end_min = min(29.0, max_time)

        time_min = df_sprint_quali['Time'].dt.total_seconds() / 60

        df_sprint_quali['Quali_Session'] = np.select(
            [
                time_min <= sq1_end_min,
                (time_min > sq1_end_min) & (time_min <= sq2_end_min),
                time_min > sq2_end_min
            ],
            ['SQ1', 'SQ2', 'SQ3'],
            default='Unknown'
        )

        df_sprint_quali = df_sprint_quali[df_sprint_quali['Quali_Session'] != 'Unknown'].copy()

        sq3_rain_mask = (
            (df_sprint_quali['Quali_Session'] == 'SQ3') &
            (df_sprint_quali['Rainfall'] == 1)
        )

        if sq3_rain_mask.sum() > 10:
            df_sprint_quali = df_sprint_quali[
                (df_sprint_quali['Quali_Session'] != 'SQ3') | sq3_rain_mask
            ].copy()

        sprint_fast = df_sprint_quali.groupby(
            ['Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session',
             'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp']
        )['laptime_sum_sectortimes'].min().reset_index()

        if not sprint_fast.empty:
            sprint_fast['LapTimeDiff'] = (
                sprint_fast.groupby('Quali_Session')['laptime_sum_sectortimes']
                .transform(lambda s: s - s.min())
            )

            merge_cols = [
                'Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session',
                'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
                'laptime_sum_sectortimes'
            ]

            q_fast_time = pd.merge(
                sprint_fast,
                df_sprint_quali.loc[:, merge_cols + ['Compound']],
                on=merge_cols,
                how='left'
            )

            fastest_idx = q_fast_time.groupby(
                ['Driver', 'Team', 'GP', 'Rule_Era', 'Quali_Session']
            )['laptime_sum_sectortimes'].idxmin()

            q_fast_time = q_fast_time.loc[fastest_idx].reset_index(drop=True)

            session_order = {'SQ1': 1, 'SQ2': 2, 'SQ3': 3}
            q_fast_time['session_rank'] = q_fast_time['Quali_Session'].map(session_order)

            deepest_idx = q_fast_time.groupby(
                ['Driver', 'Team', 'GP', 'Rule_Era']
            )['session_rank'].idxmax()

            df_sprint_quali_overview = (
                q_fast_time.loc[deepest_idx]
                .drop(columns='session_rank')
                .rename(columns={'Quali_Session': 'Sprint-Session'})
                .sort_values('laptime_sum_sectortimes')
                .reset_index(drop=True)
            )

            group_col = ['Driver', 'Team', 'GP', 'Rule_Era', 'Sprint-Session']
            sprint_col = [c for c in df_sprint_quali_overview.columns if c not in group_col]

            df_sprint_quali_overview = df_sprint_quali_overview.rename(
                columns={qc: f'{qc}_sprint_quali' for qc in sprint_col}
            )

    # -----------------------------
    # FINAL KEY TABLE: union of all available overview tables
    # -----------------------------
    all_key_frames = [
        df_training_overview[['Driver', 'Team', 'GP', 'Rule_Era']]
    ]

    if df_sprint_quali_overview is not None:
        all_key_frames.append(
            df_sprint_quali_overview[['Driver', 'Team', 'GP', 'Rule_Era']]
        )

    all_keys_final = (
        pd.concat(all_key_frames, ignore_index=True)
        .drop_duplicates()
        .copy()
    )

    # -----------------------------
    # sprint placeholder on final key table
    # -----------------------------
    df_sprint_quali_na = all_keys_final.copy()
    df_sprint_quali_na['Sprint-Session'] = np.nan
    for col in [
        'AirTemp_sprint_quali', 'Humidity_sprint_quali', 'Pressure_sprint_quali', 'Rainfall_sprint_quali',
        'TrackTemp_sprint_quali', 'laptime_sum_sectortimes_sprint_quali', 'LapTimeDiff_sprint_quali', 'Compound_sprint_quali'
    ]:
        df_sprint_quali_na[col] = np.nan

    df_sprint_quali_na = df_sprint_quali_na[
        ['Driver', 'Team', 'GP', 'Rule_Era', 'Sprint-Session',
         'AirTemp_sprint_quali', 'Humidity_sprint_quali', 'Pressure_sprint_quali', 'Rainfall_sprint_quali',
         'TrackTemp_sprint_quali', 'laptime_sum_sectortimes_sprint_quali', 'LapTimeDiff_sprint_quali', 'Compound_sprint_quali']
    ].copy()

    # -----------------------------
    # FINAL MERGE
    # -----------------------------
    df_final_gp = pd.merge(
        all_keys_final,
        df_training_overview,
        on=['Driver', 'Team', 'GP', 'Rule_Era'],
        how='left'
    )

    if df_sprint_quali_overview is not None:
        df_final_gp = pd.merge(
            df_final_gp,
            df_sprint_quali_overview,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )
    else:
        df_final_gp = pd.merge(
            df_final_gp,
            df_sprint_quali_na,
            on=['Driver', 'Team', 'GP', 'Rule_Era'],
            how='left'
        )

    df_final_gp['Season'] = season

    # -----------------------------
    # ERA FLAGS
    # -----------------------------
    df_final_gp['Sprint_Race_Era'] = df_final_gp['Season'].apply(
        lambda x: '2021-2022' if 2021 <= x <= 2022
        else ('2024' if x == 2024 else '2023-2026')
    )

    df_final_gp['Sprint_Weekend'] = df_final_gp['Sprint-Session'].apply(
        lambda x: 0 if pd.isna(x) else 1
    )

    return df_final_gp

def tune_offset_shrinkage_expanding(
    calibration_df,
    shrinking_values=None,
    gp_col="gp_id",
    driver_col="Driver",
    team_col="Team",
    residual_col="resid",
    prediction_col="pred",
    actual_col="actual",
):
    """
    Tune one shrinking value using expanding-window validation.

    GP2 uses GP1 offsets.
    GP3 uses GP1-GP2 offsets.
    ...
    """

    #if we do not provide a shrinking range, it will us the default range
    if shrinking_values is None:
        shrinking_values = np.arange(0.0, 1.01, 0.05)

    #here we will get the unique values for all avaiable gp_ids in the dataframe
    gp_ids = np.sort(calibration_df[gp_col].unique())
    results = []

    #here we loop different shrinking_values
    for shrinking in shrinking_values:
        expanding_predictions = []

        #start with GP2 because GP1 has no previous 2026 information
        #to avoid using GP1 in the validation, we start the range with 1
        for i in range(1, len(gp_ids)):
            #making sure to include everything up to the range value
            previous_gp_ids = gp_ids[:i]
            current_gp_id = gp_ids[i]

            history = calibration_df[
                calibration_df[gp_col].isin(previous_gp_ids)
            ]

            current_gp = calibration_df[
                calibration_df[gp_col].eq(current_gp_id)
            ].copy()

            global_offset = history[residual_col].median()

            #creating the different offsets
            driver_offsets = (
                history.groupby(driver_col)[residual_col].median()
            )

            team_offsets = (
                history.groupby(team_col)[residual_col].median()
            )

            current_gp["driver_offset"] = (
                current_gp[driver_col]
                .map(driver_offsets)
                .fillna(global_offset)
            )

            current_gp["team_offset"] = (
                current_gp[team_col]
                .map(team_offsets)
                .fillna(global_offset)
            )

            #adding the offsets to the current GP in the expanding window
            current_gp["gp_global_offset"] = global_offset

            #taking the mean to get my offset, that will be applied to the prediction
            current_gp["offset"] = current_gp[
                [
                    "driver_offset",
                    "team_offset",
                    "gp_global_offset",
                ]
            ].mean(axis=1)

            current_gp["pred_corrected"] = (
                current_gp[prediction_col]
                + shrinking * current_gp["offset"]
            )

            expanding_predictions.append(current_gp)

        expanding_df = pd.concat(
            expanding_predictions,
            ignore_index=True,
        )

        mae = mean_absolute_error(
            expanding_df[actual_col],
            expanding_df["pred_corrected"],
        )

        results.append(
            {
                "shrinking": shrinking,
                "mae": mae,
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("mae")
        .reset_index(drop=True)
    )

    best_shrinking = results_df.loc[0, "shrinking"]

    return best_shrinking, results_df

def df_for_prediction(
        df_circut = str,
        quali_data = str,
        df_filter = ['British'],

):

    f1_calendar_by_season = {
        2018: [
            "Australian GP", "Bahrain GP", "Chinese GP", "Azerbaijan GP",
            "Spanish GP", "Monaco GP", "Canadian GP", "French GP",
            "Austrian GP", "British GP", "German GP", "Hungarian GP",
            "Belgian GP", "Italian GP", "Singapore GP", "Russian GP",
            "Japanese GP", "United States GP", "Mexican GP", "Brazilian GP",
            "Abu Dhabi GP",
        ],

        2019: [
            "Australian GP", "Bahrain GP", "Chinese GP", "Azerbaijan GP",
            "Spanish GP", "Monaco GP", "Canadian GP", "French GP",
            "Austrian GP", "British GP", "German GP", "Hungarian GP",
            "Belgian GP", "Italian GP", "Singapore GP", "Russian GP",
            "Japanese GP", "Mexican GP", "United States GP", "Brazilian GP",
            "Abu Dhabi GP",
        ],

        2020: [
            "Austrian GP", "Styrian GP", "Hungarian GP", "British GP",
            "70th Anniversary GP", "Spanish GP", "Belgian GP", "Italian GP",
            "Tuscan GP", "Russian GP", "Eifel GP", "Portuguese GP",
            "Emilia Romagna GP", "Turkish GP", "Bahrain GP", "Sakhir GP",
            "Abu Dhabi GP",
        ],

        2021: [
            "Bahrain GP", "Emilia Romagna GP", "Portuguese GP", "Spanish GP",
            "Monaco GP", "Azerbaijan GP", "French GP", "Styrian GP",
            "Austrian GP", "British GP", "Hungarian GP", "Belgian GP",
            "Dutch GP", "Italian GP", "Russian GP", "Turkish GP",
            "United States GP", "Mexican GP", "Brazilian GP", "Qatar GP",
            "Saudi Arabian GP", "Abu Dhabi GP",
        ],

        2022: [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Emilia Romagna GP",
            "Miami GP", "Spanish GP", "Monaco GP", "Azerbaijan GP",
            "Canadian GP", "British GP", "Austrian GP", "French GP",
            "Hungarian GP", "Belgian GP", "Dutch GP", "Italian GP",
            "Singapore GP", "Japanese GP", "United States GP", "Mexican GP",
            "Brazilian GP", "Abu Dhabi GP",
        ],

        2023: [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Azerbaijan GP",
            "Miami GP", "Monaco GP", "Spanish GP", "Canadian GP",
            "Austrian GP", "British GP", "Hungarian GP", "Belgian GP",
            "Dutch GP", "Italian GP", "Singapore GP", "Japanese GP",
            "Qatar GP", "United States GP", "Mexican GP", "Brazilian GP",
            "Las Vegas GP", "Abu Dhabi GP",
        ],

        2024: [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Japanese GP",
            "Chinese GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
            "Canadian GP", "Spanish GP", "Austrian GP", "British GP",
            "Hungarian GP", "Belgian GP", "Dutch GP", "Italian GP",
            "Azerbaijan GP", "Singapore GP", "United States GP", "Mexican GP",
            "Brazilian GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
        ],

        2025: [
            "Australian GP", "Chinese GP", "Japanese GP", "Bahrain GP",
            "Saudi Arabian GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
            "Spanish GP", "Canadian GP", "Austrian GP", "British GP",
            "Belgian GP", "Hungarian GP", "Dutch GP", "Italian GP",
            "Azerbaijan GP", "Singapore GP", "United States GP", "Mexican GP",
            "Brazilian GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
        ],

        #2026 was already adjusted due to the canceld Saudi Arabia and Barhain GPs
        2026: [
            "Australian GP", "Chinese GP", "Japanese GP", "Miami GP",
            "Canadian GP", "Monaco GP", "Spanish GP", "Austrian GP",
            "British GP", "Belgian GP", "Hungarian GP", "Dutch GP",
            "Italian GP", "Madrid GP", "Azerbaijan GP", "Singapore GP",
            "United States GP", "Mexican GP", "Brazilian GP", "Las Vegas GP",
            "Qatar GP", "Abu Dhabi GP",
        ],
    }

    #df_circut = pd.read_csv(DATA_PROCESSED / 'cleaned_f1_circut.csv')
    df_circut = pd.read_csv(df_circut)

    df_2018_2026 = pd.DataFrame()
    start_range = 2018
    end_range = 2026

    for season in range(start_range,end_range+1):
        #turning "season" to string, because the path cannot processes a numerical value
        #data_path = DATA_RAW / str(season)
        data_path = quali_data/str(season)

        #getting name of folder if the folder exists
        path_list = sorted([f.name for f in data_path.iterdir() if f.is_dir() and not f.name.startswith('.') and f.name.endswith("GP")])

        # keep this only if you really need it
        #if season <= 2026:
        #path_list = path_list[1:]

        if season == 2020:
            path_list = [x for x in path_list if x != "Eifel GP"]

        #entering each folder of a Grand Prix weekend
        for folder_loop_value in path_list:
            base = data_path / folder_loop_value

            timing_paths, weather_paths = build_session_paths(
                base=base,
                season=season,
                folder_loop_value=folder_loop_value
            )

            df_loop = practice_quali_new_pred(
                season=season,
                gp=folder_loop_value,
                timing_paths=timing_paths,
                weather_paths=weather_paths,
                w_vis=False
            )

            df_2018_2026 = pd.concat([df_2018_2026, df_loop], ignore_index=True)
    #creating the mapping => key: season, gp; value: number of the gp-weekend
    race_round_map = {
        (season, gp): round_no
        for season, races in f1_calendar_by_season.items()
        for round_no, gp in enumerate(races, start=1)
    }

    #race_round will be used as an X-feature
    df_2018_2026["race_round"] = df_2018_2026.apply(
        lambda row: race_round_map.get((row["Season"], row["GP"])),
        axis=1
    )

    #creating a dataframe to create an race order
    weekend_order = (
        df_2018_2026[["Season", "GP", "race_round"]]
        .drop_duplicates()
        .sort_values(["Season", "race_round"])
        .reset_index(drop=True)
    )
    #creating a unique id based on the race weekend order
    #group_id will be used for time-aware cross-validation
    weekend_order["gp_id"] = range(len(weekend_order))

    #finally joining the race order logic
    df_2018_2026 = df_2018_2026.merge(
        weekend_order[["Season", "GP", "gp_id"]],
        on=["Season", "GP"],
        how="left"
    )

    team_lineage_map = {
        "Toro Rosso": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
        "AlphaTauri": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
        "RB": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
        "Racing Bulls": "Toro Rosso-AlphaTauri-RB-Racing Bulls",

        "Sauber": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Alfa Romeo": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Alfa Romeo Racing": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Kick Sauber": "Sauber-Alfa Romeo-Kick Sauber-Audi",
        "Audi": "Sauber-Alfa Romeo-Kick Sauber-Audi",

        "Force India":"Force India-Racing Point-Aston Martin",
        "Racing Point":"Force India-Racing Point-Aston Martin",
        "Aston Martin":"Force India-Racing Point-Aston Martin",
        "Alpine":"Alpine-Renault",
        "Renault":"Alpine-Renault"
    }

    df_2018_2026["Team_Lineage"] = df_2018_2026["Team"].replace(team_lineage_map)
    df_quali = pd.merge(df_2018_2026,df_circut,how='left',on='GP')
    df_quali["Pace_profile"] = df_quali["Pace_profile"].str.strip()
    current_season = df_quali['Season'].max()
    df_quali = df_quali[df_quali['Season'].eq(current_season)]
    df_quali = df_quali[(df_quali['laptime_sum_sectortimes_p2'].notna()) | (df_quali['laptime_sum_sectortimes_sprint_quali'].notna())].copy()

    #determining the Grand Prix
    ##checking if the value is None; None will trigger the prediction of all available data for the current season
    if df_filter is None:
        current_season = df_quali['Season'].max()
        df_quali = df_quali[df_quali['Season'].eq(current_season)].copy()
    else:
        df_filter_gp = [gp+' GP' for gp in df_filter]
        gp_f = df_filter_gp
        #using the most recent data of the selected Grand Prix
        current_season = df_quali['Season'].max()
        df_quali = df_quali[(df_quali['GP'].isin(gp_f)) & (df_quali['Season'].eq(current_season))].copy()

    return df_quali

# ---------------------------------------------- Data Drift ----------------------------------------------
def check_data_drift(
        X_old,
        X_new,
        drift_share=0.92):

    binary_cols = [
        col for col in X_old.select_dtypes(exclude="object").columns
        if set(X_old[col].dropna().unique()).issubset({0, 1})
    ]

    numeric_cols = [
        col for col in X_old.select_dtypes(exclude="object").columns
        if col not in binary_cols
    ]

    categorical_cols = list(
        X_old.select_dtypes(include="object").columns
    )

    categorical_cols = categorical_cols + binary_cols

    schema = DataDefinition(
        numerical_columns=numeric_cols,
        categorical_columns=categorical_cols
    )

    reference_data = Dataset.from_pandas(
        X_old[numeric_cols + categorical_cols],
        data_definition=schema
    )

    current_data = Dataset.from_pandas(
        X_new[numeric_cols + categorical_cols],
        data_definition=schema
    )

    report = Report([
        DataDriftPreset(
            drift_share=drift_share
        )
    ])

    result = report.run(
        current_data=current_data,
        reference_data=reference_data
    )

    return result.dict()

def make_drift_table(drift_result):

    rows = []

    for metric in drift_result["metrics"]:

        config = metric["config"]

        # We only want individual feature drift metrics
        if config["type"] != "evidently:metric_v2:ValueDrift":
            continue

        feature = config["column"]
        method = config["method"]
        threshold = config["threshold"]
        score = metric["value"]

        # p-value based tests:
        # smaller score = stronger evidence of drift
        if "p_value" in method.lower():
            drift_detected = score <= threshold

        # distance / divergence based methods:
        # larger score = more drift
        else:
            drift_detected = score >= threshold

        rows.append({
            "feature": feature,
            "method": method,
            "score": score,
            "threshold": threshold,
            "drift_detected": drift_detected
        })

    return pd.DataFrame(rows)

def get_drift_summary(drift_result):

    overall = next(
        metric for metric in drift_result["metrics"]
        if metric["config"]["type"] ==
        "evidently:metric_v2:DriftedColumnsCount"
    )

    drifted_count = int(overall["value"]["count"])
    drift_share = overall["value"]["share"]
    drift_threshold = overall["config"]["drift_share"]

    total_features = round(drifted_count / drift_share)

    return pd.DataFrame({
        "Metric": [
            "Total features",
            "Drifted features",
            "Drift share",
            "Dataset drift threshold",
            "Overall drift detected"
        ],
        "Value": [
            total_features,
            drifted_count,
            f"{drift_share:.1%}",
            f"{drift_threshold:.1%}",
            drift_share >= drift_threshold
        ]
    })

def get_drift_share(drift_result):
    """Extract overall drift share from Evidently result."""

    for metric in drift_result["metrics"]:
        if metric["config"]["type"] == "evidently:metric_v2:DriftedColumnsCount":
            return {
                "drifted_features": int(metric["value"]["count"]),
                "drift_share": metric["value"]["share"]
            }

    raise ValueError("DriftedColumnsCount not found")

def historical_drift_backtest(
    X,
    start_year=2021,
    max_round=14,
    exclude_cols=None
):

    if exclude_cols is None:
        exclude_cols = []

    results = []

    years = sorted(X["Season"].unique())

    for year in years:

        if year < start_year:
            continue

        # Everything BEFORE this season = historical reference
        X_reference = X[X["Season"] < year].copy()

        # Pretend this season is the "new" data
        # Only use the first N rounds, matching your current 2026 situation
        X_current = X[
            (X["Season"] == year) &
            (X["race_round"] <= max_round)
        ].copy()

        if len(X_reference) == 0 or len(X_current) == 0:
            continue

        # Don't allow purely temporal / ID variables
        # to dominate the drift decision
        reference_drift = X_reference.drop(
            columns=exclude_cols,
            errors="ignore"
        )

        current_drift = X_current.drop(
            columns=exclude_cols,
            errors="ignore"
        )

        drift_result = check_data_drift(
            X_old=reference_drift,
            X_new=current_drift
        )

        summary = get_drift_share(drift_result)

        results.append({
            "year": year,
            "reference_years":
                f"{int(X_reference['Season'].min())}-{year-1}",
            "current_rounds": f"1-{max_round}",
            "n_reference": len(X_reference),
            "n_current": len(X_current),
            "drifted_features": summary["drifted_features"],
            "drift_share": summary["drift_share"]
        })

    return pd.DataFrame(results)

def historical_drift_backtest_transformed(
    X_raw,
    fitted_pipeline,
    start_year=2021,
    max_round=14
):

    # ---------------------------------
    # Transform using FITTED pipeline
    # ---------------------------------
    step_names = list(fitted_pipeline.named_steps.keys())
    selector_idx = step_names.index("Feature_Selector")

    # Everything before selector => here we perform 
    X_encoded = fitted_pipeline[:selector_idx].transform(X_raw)

    # Apply fitted selector
    selector = fitted_pipeline["Feature_Selector"]
    X_selected = selector.transform(X_encoded)

    encoded_cols = X_selected.columns.difference(X_raw.columns,sort=False)
    X_selected = X_selected[encoded_cols].copy()

    results = []

    years = sorted(X_raw["Season"].unique())

    for year in years:

        if year < start_year:
            continue

        reference_idx = X_raw.index[
            X_raw["Season"] < year
        ]

        current_idx = X_raw.index[
            (X_raw["Season"] == year) &
            (X_raw["race_round"] <= max_round)
        ]

        if len(reference_idx) == 0 or len(current_idx) == 0:
            continue

        X_reference = X_selected.loc[reference_idx]
        X_current = X_selected.loc[current_idx]

        drift_result = check_data_drift(
            X_old=X_reference,
            X_new=X_current,
            drift_share=0.5
        )

        summary = get_drift_share(drift_result)

        results.append({
            "year": year,
            "reference_years":
                f"{int(X_raw.loc[reference_idx, 'Season'].min())}-{year-1}",
            "current_rounds": f"1-{max_round}",
            "n_reference": len(X_reference),
            "n_current": len(X_current),
            "drifted_features": summary["drifted_features"],
            "drift_share": summary["drift_share"]
        })

    return pd.DataFrame(results)

def historical_drift_backtest_target(
    X_data_for_year,
    y,
    start_year=2021,
    max_round=14,
    exclude_cols=None
):

    if exclude_cols is None:
        exclude_cols = []

    results = []

    years = sorted(X_data_for_year["Season"].unique())

    for year in years:

        if year < start_year:
            continue

        # Everything BEFORE this season = historical reference
        X_reference = X_data_for_year[X_data_for_year["Season"] < year].copy()

        # Pretend this season is the "new" data
        # Only use the first N rounds, matching your current 2026 situation
        X_current = X_data_for_year[
            (X_data_for_year["Season"] == year) &
            (X_data_for_year["race_round"] <= max_round)
        ].copy()

        if len(X_reference) == 0 or len(X_current) == 0:
            continue

        year_index_ref = X_reference.index
        year_index_current = X_current.index
        

        # Don't allow purely temporal / ID variables
        # to dominate the drift decision
        reference_drift = y.loc[year_index_ref]

        current_drift = y.loc[year_index_current]

        drift_result = check_data_drift(
            X_old=reference_drift,
            X_new=current_drift
        )

        summary = get_drift_share(drift_result)

        results.append({
            "year": year,
            "reference_years":
                f"{int(X_reference['Season'].min())}-{year-1}",
            "current_rounds": f"1-{max_round}",
            "n_reference": len(X_reference),
            "n_current": len(X_current),
            "drifted_features": summary["drifted_features"],
            "drift_share": summary["drift_share"]
        })

    return pd.DataFrame(results)