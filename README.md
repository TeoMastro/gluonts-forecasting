## Installation

To install TSB-Forecasting from source, you will need the following tools:
- `git`
- `conda` (anaconda or miniconda)

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/TheDatumOrg/TSB-Forecasting.git
```

**Step 2:** Create and activate a `conda` environment from the env.yaml file that is provided.

```bash
conda env create -f env.yml
conda activate tsb-ad
```

**Step 4:** Open your editor of choice and start coding.

In case you want to install with pip, the requirements.txt is also updated, although I highly recommend anaconda.

## To contribute to the library
**Step 1:** Create/modify a model class inside the models folder.

**Step 2:** Create/modify a wrapper function in model_wrapper.py (for univariate forecasting) or in multivariate_model_wrapper.py (for multivariate forecasting).
The workflow that these wrapper functions have is the following:
1. Get/parse data from a dataset (either univariate or multivariate depending on the wrapper file you are on). Use one of the existing data parsers or create your own based on your model's needs.
2. Import and initialize a model.
3. Use the corrensponding .fit() function of the model.
4. Use the corrensponding .predict() function of the model.
5. Generate some metrics (MAE, MSE, RMSE, R2), as we have them for the other models.
6. Go in the main.py file, and test yor model by calling its wrapper function (like the provided example).

**Step 3:** Push to a new branch. 
1. Make sure you are on main.
2. Do `git pull` on main.
3. Make sure you create a branch using:
```bash
git checkout -b my-branch-name
```
4. `git add .`
5. `git commit -m "Added ModelXYZ"`
6. `git push -u origin my-branch-name`

After you do this, go back to main and create another branch in order to add another model.

**SOS:** Before you upload a branch, make sure you have the latest main. Then go back to your branch and do `git rebase` with the main branch, so your branch is always one step ahead of the main.

## Contact
If you have any questions or suggestions, feel free to contact:
* Theodoros Matromanolis (tmast@csd.auth.gr)

Or describe it in Issues.