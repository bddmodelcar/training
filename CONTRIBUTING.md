# Contributing
Contributions should be made using the GitHub Standard Pull Request Workflow. To set this up do the following:
1. Fork this repository on GitHub (press the fork button)
2. Clone your forked repository (git clone ...)
3. Add this repository as upstream

```
git remote add upstream https://github.com/bddmodelcar/training.git
```

4. Make your changes, commit and push locally to your fork of the repository.
5. Sync changes to the remote repository (Should be done intermittently throughout work)

```
git fetch upstream
git merge upstream/master
```

6. When you are ready to push your changes globally, open your fork in GitHub and press the Pull Request button. 
If you are a collaborator, you can merge the pull request yourself, but it is suggested to let others review your 
changes and merge.


# Save Files
Save files are managed by the Git LFS system, any file that ends with .infer or .weights will be managed by this system. 
If you have not installed Git LFS, these save files will not be downloaded and a plain text file will exist in their 
place. If you wish to create or use some of these save files you can do the following:

1. Download and install the [Git LFS command line extension](https://github.com/git-lfs/git-lfs/releases/download/v2.2.1/git-lfs-linux-amd64-2.2.1.tar.gz)
2. `git lfs install`
3. `echo "export GIT_LFS_SKIP_SMUDGE=1" >> ~/.bashrc`
4. `source ~/.bashrc`
5. Work normally
6. If you want to download GIT LFS files rather than having pointers than you can run `git lfs pull`
