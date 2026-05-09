from fastapi import FastAPI, HTTPException
import asyncio
import os

app = FastAPI()

# Define the mapping of domain names to their reset scripts
DOMAIN_SCRIPTS = {
    "shopping": "shopping-reset.sh",
    "shopping_admin": "shopping-admin-reset.sh",
    "reddit": "reddit-reset.sh",
    "gitlab": "gitlab-reset.sh",
}

@app.post("/reset/{domain}")
async def reset_domain(domain: str):
    # Map does not require reset as it is stateless
    if domain == "map":
        return {"message": "Map domain does not require reset."}
    
    if domain == "all":
        # Reset all domains
        results = {}
        for domain_name, script_path in DOMAIN_SCRIPTS.items():
            if os.path.exists(script_path):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "/bin/bash", script_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    
                    if proc.returncode == 0:
                        results[domain_name] = {
                            "message": f"{domain_name} reset successfully", 
                            "output": stdout.decode()
                        }
                    else:
                        results[domain_name] = {
                            "error": f"Error resetting {domain_name}: {stderr.decode()}"
                        }
                except Exception as e:
                    results[domain_name] = {"error": f"Exception while resetting {domain_name}: {str(e)}"}
            else:
                results[domain_name] = {"error": f"Reset script not found for {domain_name}"}
        return results

    if domain not in DOMAIN_SCRIPTS:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    script_path = DOMAIN_SCRIPTS[domain]
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=500, detail=f"Reset script not found for {domain}")
    
    try:
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            return {"message": f"{domain} reset successfully", "output": stdout.decode()}
        else:
            raise HTTPException(status_code=500, detail=f"Error resetting {domain}: {stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception while resetting {domain}: {str(e)}")