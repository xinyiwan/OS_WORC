import paramiko
import logging
import subprocess
from scp import SCPClient
import os

def test_ssh_connection(server, username, port=22):
    """Test SSH connection to remote server"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load SSH config from default location
        ssh_config = paramiko.SSHConfig()
        user_config_file = os.path.expanduser("~/.ssh/config")
        if os.path.exists(user_config_file):
            with open(user_config_file) as f:
                ssh_config.parse(f)

        # Get configuration for the host
        host_config = ssh_config.lookup(server)
        
        # Use parameters from SSH config
        connect_kwargs = {
            'hostname': host_config.get('hostname', server),
            'port': int(host_config.get('port', port)),
            'username': host_config.get('user', username),
        }
        
        # Add identity file if specified
        if 'identityfile' in host_config:
            connect_kwargs['key_filename'] = host_config['identityfile']
        
        ssh.connect(**connect_kwargs)
        ssh.close()
        return True
    except Exception as e:
        logging.error(f"SSH connection failed: {e}")
        return False
    


def copy_with_rsync(local_path, remote_path, server, username, port=22):
    """Copy files using rsync (more efficient for large files)"""
    try:
        # Create remote directory first
        remote_dir = os.path.dirname(remote_path)
        ssh_cmd = f"ssh -p {port} {username}@{server} 'mkdir -p {remote_dir}'"
        subprocess.run(ssh_cmd, shell=True, check=True)
        
        # Use rsync for transfer
        rsync_cmd = [
            'rsync', '-avz', '--progress', '-e', f'ssh -p {port}',
            local_path, f'{username}@{server}:{remote_path}'
        ]
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Successfully copied {local_path} to {server}:{remote_path}")
            return True
        else:
            logging.error(f"rsync failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"rsync process error: {e}")
        return False
    except Exception as e:
        logging.error(f"rsync unexpected error: {e}")
        return False

def copy_with_scp(local_path, remote_path, server, username, port=22):
    """Copy files using SCP (alternative method)"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server, port=port, username=username)
        
        # Create remote directory
        remote_dir = os.path.dirname(remote_path)
        stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_dir}')
        stdout.channel.recv_exit_status()  # Wait for command to complete
        
        # Copy file
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path)
        
        ssh.close()
        logging.info(f"Successfully copied {local_path} to {server}:{remote_path}")
        return True
        
    except Exception as e:
        logging.error(f"SCP failed for {local_path}: {e}")
        return False

def copy_with_subprocess_scp(local_path, remote_path, server, username, port=22):
    """Copy files using subprocess SCP (simple alternative)"""
    try:
        # Create remote directory first
        remote_dir = os.path.dirname(remote_path)
        mkdir_cmd = f"ssh -p {port} {username}@{server} 'mkdir -p {remote_dir}'"
        subprocess.run(mkdir_cmd, shell=True, check=True)
        
        # Copy file
        scp_cmd = f"scp -P {port} {local_path} {username}@{server}:{remote_path}"
        result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Successfully copied {local_path} to {server}:{remote_path}")
            return True
        else:
            logging.error(f"SCP failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"SCP process error: {e}")
        return False
    except Exception as e:
        logging.error(f"SCP unexpected error: {e}")
        return False