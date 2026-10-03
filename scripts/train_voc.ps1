param([Parameter(Mandatory=$true)][string]$DataRoot)
python tools/train.py --config configs/voc.yaml --data-root $DataRoot
