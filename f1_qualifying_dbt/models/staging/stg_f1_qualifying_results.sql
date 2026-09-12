{{config(
    materialized = 'incremental',
    incremental_strategy = 'append'
)}}

with raw as (
    Select *
    from {{ ref('Cleaned_quali_f1') }}
),
final as (
    Select
        Driver,
        Team,
        GP,
        Rule_Era,
        AirTemp_p1,
        Humidity_p1,
       Pressure_p1,
       Rainfall_p1,
       laptime_sum_sectortimes_p1,
       LapTimeDiff_p1,
       Compound_p1,
       AirTemp_p2,
       Humidity_p2,
       Pressure_p2,
       Rainfall_p2,
       laptime_sum_sectortimes_p2,
       LapTimeDiff_p2,
       Compound_p2,
       AirTemp_p3,
       Humidity_p3,
       Pressure_p3,
       Rainfall_p3,
       laptime_sum_sectortimes_p3,
       LapTimeDiff_p3,
       Compound_p3,
       "Sprint-Session" as Sprint_Session,
       AirTemp_sprint_quali,
       Humidity_sprint_quali,
       Pressure_sprint_quali,
       Rainfall_sprint_quali,
       laptime_sum_sectortimes_sprint_quali,
       LapTimeDiff_sprint_quali,
       Compound_sprint_quali,
       Session,
       AirTemp_quali,
       Humidity_quali,
       Pressure_quali,
       Rainfall_quali,
       laptime_sum_sectortimes_quali,
       LapTimeDiff_quali,
       Compound_quali,
       Season,
       Sprint_Race_Era,
       Sprint_Weekend,
       race_round,
       gp_id,
       Team_Lineage,
       Type,
       Direction,
       Circut_length,
       Turns,
       Pace_profile,
       Flat_out_run,
       Slow_turns,
       Medium_turns,
       High_speed_turns,
       Turn_density,
       Complexity_label,
       cast(Qualifying_Date as date) as Qualifying_Date
    from raw
)

Select * from final

{% if is_incremental() %}

where Qualifying_Date > (
    select max(Qualifying_Date)
    from {{ this }}
)

{% endif %}