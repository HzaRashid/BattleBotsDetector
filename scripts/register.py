from argparse import ArgumentParser
import requests
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument("--root-url") # http://[ip-address]:[port]
    parser.add_argument("--key") # JWT

    # register as a bot or detector
    parser.add_argument("--bot", action="store_true")
    parser.add_argument("--detector", action="store_true")

    # github repo info
    parser.add_argument("-r", "--git-repo", "--git-url") # https://github.com/[git-username]/[git-repo-name]
    parser.add_argument("-b", "--git-branch")

    # openai api key
    parser.add_argument("--openai-key")

    args = parser.parse_args()

    user_objective = "bot" if args.bot else "detector" if args.detector else None 
    if not user_objective: 
        print("error: bot or detector not specified \
              \nhint: use flag \"--bot\" or \"--detector\"")
        return
    
    api = f'{args.root_url}/api' if args.root_url else os.getenv('BASE_URL_V2')
    if not api:
        print("error: no host url")
        return
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {args.key or os.getenv('AUTH_TOKEN')}",
        "Content-Type": "application/json",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
    
    data = {
        "github_url": args.git_repo or os.getenv('GITHUB_URL'),
        "github_branch": args.git_branch or os.getenv('GITHUB_BRANCH'),
        "env_var1": args.openai_key or "string",
        "env_var2": "string"
        }
    registration_url = f"{api}/{user_objective}/session/{args.session_id}/register"
    
    if not (data["github_url"] and data["github_branch"]):
        print("error: github url or branch not specified")
        return
    
    print(f"\nregistering \"{data["github_branch"]}\" branch of \"{data["github_url"]}\" to session {args.session_id}...\n")
    
    response = requests.post(
        url=registration_url, 
        headers=headers, 
        json=data)
    
    print(f"status: {response.status_code}\n")
    print(f"message: {response.content.decode('ascii')}\n")
    # print(f"headers: {response.headers}\n")
    
    
if __name__ == "__main__":
    main()