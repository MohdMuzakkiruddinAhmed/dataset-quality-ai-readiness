# Dataset sources

No row-level dataset is redistributed in this repository. The saved aggregate evidence uses three public UCI sources. UCI lists all three under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain their attribution and identify project transformations when reusing the data.

| Dataset | Source and attribution | Evaluated transformation |
|---|---|---|
| Adult | Barry Becker and Ronny Kohavi (1996), [UCI record](https://archive.ics.uci.edu/dataset/2/adult), [DOI](https://doi.org/10.24432/C5XW20) | Normalized complete cases; exact duplicate exclusion; feature-group-disjoint stratified split. |
| Diabetes 130-US Hospitals for Years 1999–2008 | John Clore, Krzysztof Cios, Jon DeShazo, and Beata Strack (2014), [UCI record](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [DOI](https://doi.org/10.24432/C5230J) | Eligible encounters; recorded 30-day readmission; retained natural missingness; patient-disjoint split. |
| Steel Industry Energy Consumption | Sathishkumar V E, Changsun Shin, and Yongyun Cho (2021), [UCI record](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), [DOI](https://doi.org/10.24432/C52G8C) | Within-day lagged predictors; recorded Load_Type classification; chronological holdout. |

The implemented source URLs, checksums, schema, and transformations are retained in `src/dataset_readiness/datasets/`. These functions can acquire public inputs deliberately after installing the toolkit:

```python
from pathlib import Path
from dataset_readiness.datasets.adult import download_adult
from dataset_readiness.datasets.diabetes import download_diabetes
from dataset_readiness.datasets.steel import download_steel

destination = Path("acquired_sources")
download_adult(destination / "adult")
download_diabetes(destination / "diabetes")
download_steel(destination / "steel")
```

Downloads are optional and are not required for the numerical replay or synthetic example. Checksum mismatches stop acquisition; metadata responses can change upstream. Preserve historical input identity rather than weakening checks.

The evaluated training/test sizes are 36,140/9,035 (Adult), 79,473/19,870 (Diabetes), and 25,662/8,648 (Steel). These are project reference populations, not the original UCI row counts. The source providers do not endorse the transformations or conclusions. The MIT license for this project does not replace the upstream dataset licenses.
