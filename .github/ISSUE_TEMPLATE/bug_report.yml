name: Bug Report
description: Create a report to help us improve.
labels:
  - Triage Pending ⚠️
body:
  - type: checkboxes
    id: latest_cl
    attributes:
      label: Are you on the latest chainladder version?
      options:
        - label: Yes, this bug occurs on the latest version.
          required: true
  - type: textarea
    id: description
    attributes:
      label: Describe the bug in words
      description: Please describe the bug you are facing clearly and concisely.
      placeholder: I am getting the wrong triangle shape returned.
    validations:
      required: true
  - type: textarea
    id: product
    attributes:
      label: How can the bug be reproduced?
      description: >-
        Steps to reproduce the behavior. Code should be self-contained and
        runnable against publicly available data.
      placeholder: |
        ```python  
        import chainladder as cl  
        triangle = cl.load_sample('raa') 
        triangle 
        ```
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: What is the expected behavior?
      description: Please use an `assert` statement.
      placeholder: |
        ```python
        assert triangle.shape == (1, 1, 10, 10)
        ```
    validations:
      required: true
