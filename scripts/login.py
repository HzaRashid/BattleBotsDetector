from argparse import ArgumentParser
import requests
import subprocess
import os

def main():
    parser = ArgumentParser()
    # competition/testing host
    parser.add_argument("--root-url") # http://[ip-address]:[port]

    # user credentials
    parser.add_argument("-u", "--username", "--teamname")
    parser.add_argument("-p", "--password") 

    # browser to open for jwt authorization
    parser.add_argument("--browser") 

    # your github repo
    parser.add_argument("-r", "--git-repo", "--git-url") # https://github.com/[git-username]/[git-repo-name]
    parser.add_argument("-b", "--git-branch")

    # if true, authorizes your JWT
    parser.add_argument('--authorize', '-A', action='store_true')

    # if true, updates your .env file
    parser.add_argument('--env', '-E', action='store_true')

    args = parser.parse_args()

    # login
    response = requests.post(
        url=f"{args.root_url}/api/auth/login", 
        headers={
            "accept": "*/*",
            "Content-Type": "application/json"
            }, 
        json={
            "team_name": args.username, 
            "team_password": args.password
        })
    
    if response.status_code != 201:
        print("login failed")
        return
    
    # fetch JWT from successful login
    jwt_key = response.content.decode('ascii')

    cur_dir = os.path.dirname(__file__)

    # authorize JWT
    if args.authorize:
        authorize_path = os.path.join(cur_dir, 'authorize.py')
        subprocess.run(
            ['python3', authorize_path,
            f'--key={jwt_key}',
            f'--api={args.root_url}/docs',
            f'--browser={args.browser}'
            ],
            )

    print(f"\nyour token 🔑: {jwt_key}\n")  

    # update env variables
    if args.env:
        env_vars = [
            f"AUTH_TOKEN={jwt_key}\n",
            f"BASE_URL={args.root_url}/api/test/1\n",
            f"SESSION_ID=1\n",
            f"MAX_TIME=3601\n",
            f"GITHUB_URL={args.git_repo or os.getenv('GITHUB_URL')}\n",
            f"GITHUB_BRANCH={args.git_branch or os.getenv('GITHUB_BRANCH')}\n",
            f"BASE_URL_V2={args.root_url}/api\n",
            f"TEAM_NAME={args.username or os.getenv('TEAM_NAME')}\n",
            f"TEAM_PASSWORD={args.password or  os.getenv('TEAM_PASSWORD')}",
        ]

        with open(os.path.join(cur_dir, '../.env'), "w+") as file:
            file.writelines(env_vars)


if __name__ == "__main__":
    main()