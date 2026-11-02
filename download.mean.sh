DEST=/glade/campaign/ral/risc/jsallen/TNC/CCKP
S3=https://wbg-cckp.s3.amazonaws.com/data

# ERA5 (2 files)
for v in tas pr; do
  mkdir -p $DEST/ERA5
  f=timeseries-$v-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc
  wget -nc -q -P $DEST/ERA5 $S3/era5-x0.25/$v/era5-x0.25-historical/$f
done

# CMIP6 historical + SSP5-8.5 (116 files)
python -c "import analog_config as c; [print(m, e) for m, e in c.MODELS.items()]" |
while read m e; do
  for v in tas pr; do
    for x in "historical 1950-2014" "ssp585 2015-2100"; do
      set -- $x
      d=$DEST/CMIP6/$v/$m-$e-$1; mkdir -p $d
      f=timeseries-$v-annual-mean_cmip6-x0.25_$m-$e-$1_timeseries_mean_$2.nc
      wget -nc -q -P $d $S3/cmip6-x0.25/$v/$m-$e-$1/$f || echo "MISSING $v $m $1"
    done
  done
done
