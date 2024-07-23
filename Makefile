# Default target
.PHONY: showMyJobs showJob sshToMachine showQueue showDir taccinfo clean

showMyJobs:
	squeue -u ysu707

showJob:
	@if [ -z "$(job_id)" ]; then \
		echo "Usage: make showJob job_id=<job_id>"; \
	else \
		squeue -j $(job_id) -o "%i %j %t %M %l %R"; \
	fi

sshToMachine:
	@if [ -z "$(machine_id)" ]; then \
		echo "Usage: make sshToMachine machine_id=<machine_id>"; \
	else \
		ssh -Y -A -o StrictHostKeyChecking=no $(machine_id); \
	fi

showQueue:
	sinfo -S+P -o "%18P %8a %20F"

showDir:
	du -a -h --max-depth=1 | sort -hr

taccinfo:
	/usr/local/etc/taccinfo

clean:
	rm -f *.e*
	rm -f *.o*
	conda clean --all
