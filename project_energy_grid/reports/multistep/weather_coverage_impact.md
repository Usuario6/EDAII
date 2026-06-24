# Impact of Historical Weather Data on Model Performance

This report records whether historical weather features overlap the E-REDES modelling window.

The historical Open-Meteo reanalysis table contains 17,544 hourly timestamps, with 17,544 overlapping E-REDES hours. Historical weather enrichment was enabled for modelling. No long-range filling or timestamp extrapolation was applied. IPMA is used separately for operational/recent weather context.

## Known Limitations

Open-Meteo is auxiliary reanalysis rather than an official Portuguese station archive, and these features describe forecast-origin conditions rather than target-time weather forecasts.
