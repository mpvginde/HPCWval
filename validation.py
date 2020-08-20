#!/usr/bin/env python3

import HPCWval

def main():
# Try setting up a validator without keywords
# HPCWval.Validator()

# Set up validator for ICON-O, medium testcase
 validator = HPCWval.Validator(model='ICON-O',testcase='medium')

# Load the 30 member ensemble with O(-16)-perturbations
 validator.loadEnsemble(members=range(1,31),dir='./O16')

# Run the statistical validator
 print('Validation of 30-member ensemble with O(-16)-perturbations:')
 validator.validate()
 print('------------------------------------')

# Load the 30 member ensemble with O(-08)-perturbations
 validator.loadEnsemble(members=range(1,31),dir='./O08')

# Run the statistical validator
 print('Validation of 30-member ensemble with O(-08)-perturbations:')
 validator.validate()
 print('------------------------------------')

if __name__ == '__main__':
  main()


  
