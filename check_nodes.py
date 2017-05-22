import pexpect as pe
from pexpect import pxssh


server_list = ['athena', 'minerva', 'neptune', 'juno', 'poseidon']
server_list2 = ['athena', 'minerva', 'neptune', 'frigga', 'odin', 'thea',
               'juno', 'thea', 'hermes', 'mercury', 'poseidon', 've', 'vili']

for i in range(len(server_list)):
    s = pxssh.pxssh()
    t = s.login(server_list[i], 'kaurov')
    t = s.sendline('uptime')  # run a command
    t = s.prompt()  # match the prompt
    temp = s.before
    nload = float(temp.split()[-1])
    t = s.sendline(r'grep processor /proc/cpuinfo | wc -l')  # run a command
    t = s.prompt()  # match the prompt
    temp = (s.before)
    nproc = float(temp.split()[-1])
    print(server_list[i], nload/nproc)
    t = s.logout()