import pysftp, pysshconfig
import os
from datetime import datetime
from getpass import getpass

with open(os.path.expanduser("~/.ssh/config")) as f:
    sshconfigs = pysshconfig.load(f)
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

_remote_dir = {
    "g1": "/home/unitree/Projects/instinct/models",
    "thinkpad": "/home/leo/Projects/instinct/models",
}

def main(args):
    """ Send the given logs to the robot and the experiment laplop. """

    # get ssh config
    sshconfig = sshconfigs.get_config_for_host(args.host)
    print("Using ssh config for host:", args.host)
    print("Host info:", sshconfig)
    cinfo = dict(
        host=sshconfig["Hostname"],
        username=sshconfig["User"],
        password=getpass("Enter password: "),
        port=int(sshconfig.get("Port", 22)),
        cnopts=cnopts,
    )

    # setup sftp connection and get files
    with pysftp.Connection(**cinfo) as sftp: # type: ignore
        print("Connected to", args.host)

        with sftp.cd(_remote_dir[args.host]):
            print("Changed to", _remote_dir[args.host])

            # get files
            print("Sending files from", args.logdir)
            if not sftp.isdir(os.path.basename(args.logdir)):
                sftp.mkdir(os.path.basename(args.logdir))
            sftp.put_r(
                args.logdir,
                os.path.join(_remote_dir[args.host], os.path.basename(args.logdir)),
            )
            print("Files sent to", args.host)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-l", type=str, required=True, help="The directory to send")
    parser.add_argument("--host", type=str, required=True, help="The machine to send to")

    args = parser.parse_args()
    main(args)
