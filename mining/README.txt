================================================================================
Mining scripts for backchannel, pause_handling, turntaking categories.
================================================================================

--------------------------------------------------------------------------------
Steps used to mine for batch1 that was provided:
--------------------------------------------------------------------------------
<language_folder> were bn_batch1 and hi_batch1
Create VADS (output will be call_wise_delivery/<language_folder>/<conversation_folder>/vads/speaker*.json):
run_silero_vad.py call_wise_delivery/<language_folder> --workers 8 --overwrite

Mine pause_handling clips (output folder will be call_wise_delivery/<language_folder>/pause_handling):
mine_pause_handling.py call_wise_delivery/<language_folder> --workers 8 --overwrite

Mine backchannel clips (output folder will be call_wise_delivery/<language_folder>/backchannel):
mine_backchannel.py call_wise_delivery/<language_folder> --workers 8 --overwrite

Mine turntaking clips (output folder will be call_wise_delivery/<language_folder>/turntaking):
mine_turntaking.py call_wise_delivery/<language_folder> --workers 8 --overwrite


--------------------------------------------------------------------------------
Steps to install python libraries:
--------------------------------------------------------------------------------
If you want to create a virtual env to install python libraries then here are the steps:
  python3.9 -m venv .venv
  source .venv/bin/activate

  python -m pip install --upgrade pip
  python -m pip install \
    silero-vad==6.2.1 \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    numpy==2.0.2 \
    soundfile==0.13.1

  Verify the environment:

  python -c "import silero_vad, torch, torchaudio, numpy, soundfile; print('Environment ready')"

  Then run scripts using either the activated environment:
  python run_silero_vad.py call_wise_delivery/bn_batch1 --workers 8 --overwrite

  Or without activation:
  .venv/bin/python run_silero_vad.py call_wise_delivery/bn_batch1 --workers 8 --overwrite
