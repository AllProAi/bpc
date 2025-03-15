# Bin-Picking Challenge Development Rules

## Overview

This project follows a set of development rules defined in the `.cursor/rules` directory. These rules are designed to ensure consistent, high-quality code and documentation throughout the project.

## Accessing the Rules

To access the rules, navigate to the `.cursor/rules` directory in the project. There you will find:

1. `README.md` - Overview and index of all rule sets
2. `general_rules.md` - General development rules
3. `python_rules.md` - Python-specific coding rules
4. `documentation_rules.md` - Documentation requirements
5. `testing_rules.md` - Testing standards and procedures
6. `process_rules.md` - Development process guidelines
7. `pose_estimator_rules.md` - Specific rules for the pose estimator implementation

## Key Development Principles

1. **Documentation First Development**
   - Begin with clear documentation before implementation
   - Keep documentation updated as code evolves

2. **Required Planning Documents**
   - Product Requirements Document (PRD)
   - Development plan
   - Technical Design Document
   - Testing strategy

3. **GitHub-Centered Workflow**
   - Develop code locally without Docker
   - Push to GitHub to trigger automated builds
   - Monitor builds through GitHub Actions

4. **Quality Standards**
   - Follow PEP 8 style guide for Python code
   - Document all code with docstrings
   - Implement comprehensive testing
   - Optimize for both accuracy and performance

## Getting Started

Before beginning development:
1. Review all rule sets in the `.cursor/rules` directory
2. Ensure all required planning documents are in place
3. Set up your development environment according to the guidelines
4. Familiarize yourself with the project structure and requirements

## Rule Precedence

When rules conflict, follow this precedence order:
1. Risk management and security rules (highest priority)
2. Process rules
3. Component-specific rules
4. Optimization rules
5. Testing rules

## Copyright

Copyright 2025, Daniel Hill dba Novus Nexum Labs (Daniel@allpro.enterprises) 