$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"

#config---->scenario path
#nodes,edges ---->result of synthetic population
#out---->output file to save the results(if it doesn't exist,its ok)

mpiexec -affinity -n 12 python "C:\path\SEAD_final.py" `
  --config "D:\CONFIGS\S0.json" `
  --nodes "D:\population\nodes_active.csv" `
  --edges-internal "D:\population\edges_internal_active.csv" `
  --edges-professional "D:\population\edges_professional_active.csv" `
  --edges-school "D:\population\edges_school_active.csv" `
  --edges-personal  "D:\population\edges_personal_facebook_active.csv" `
  --edges-family "D:\population\edges_family_active.csv" `
  --out "D:\logs\sead_timeseries_s0.csv" `
  --steps 50 `
  --seed 123


#use """   powershell -ExecutionPolicy Bypass -File "D:\Desktop\logs\run_sead.ps1"   """
#to run it in ps cmd