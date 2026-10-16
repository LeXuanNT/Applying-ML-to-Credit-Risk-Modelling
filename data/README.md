# Data

`hmeq.csv` — Home Equity (HMEQ) dataset: 5,960 home-equity loans, 13 columns (BAD + 12 application variables).

- Source: Baesens, Roesch & Scheule (2016), *Credit Risk Analytics*, Wiley — companion data (publicly distributed).
- SHA-256: `dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa`
  (checked by `python scripts/step0_data_check.py`; a different checksum means a different copy of HMEQ).
- BAD = 1: applicant defaulted or was seriously delinquent; 1,189 of 5,960 (19.95%).
- Missing values are kept as empty cells; they carry information (see thesis section 3.2.3).

Known implausible values, kept as in the thesis (see `reports/deviation_log.md`):
CLAGE > 600 months in 20 records (max 1,168), MORTDUE > VALUE in 51 records, DEBTINC > 100 in 5 records.
