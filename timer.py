import subprocess

for i in range(10):
    subprocess.run(['python', 'pilot-pipeline.py'])
    subprocess.run(['rm', '-R', 'luigi'])
    subprocess.run(['python', 'luigi-pipeline.py'])
    subprocess.run(['rm', '-R', 'luigi'])
    subprocess.run(['python', 'luigi-spark-persist-session.py'])
    subprocess.run(['rm', '-R', 'luigi'])