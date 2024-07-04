Augment your data (increase the number of rows and features)

Each dataset has to fit its own story - so the exact details must be dataset-specific...
An example of a Caesarian Section Classification Dataset is available in the main

**General Approach**:
- Restoring original features: Undo any clustering/categorization of features
- Adding features: Adding new features (correlated or independent of existing ones)
- Increase the number of samples: Duplicate and inject noise into existing samples to create more data
- Apply restriction: Ensure the added noise does not create non-sensical data (Can be combined with Nullifying data to create missing data)

Note:
  All of the steps are optional, they might not be required in your case.
  The order of the steps might be different in your case, you can apply each one and repeat as needed.
