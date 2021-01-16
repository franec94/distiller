import shlex, subprocess
cmd = "echo Hello, World!"
process = subprocess.Popen(cmd, shell=True,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)

# wait for the process to terminate
out, err = process.communicate()
errcode = process.returncode
print(out)
print(err)
print(errcode)