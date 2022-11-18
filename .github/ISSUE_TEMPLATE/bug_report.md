---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: Bug
assignees: jbogaardt

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior. Code should be self-contained and runnable against publicly available data. For example:
```python
import chainladder as cl
triangle = cl.load_sample('raa')
```

**Expected behavior**
A clear and concise description of what you expected to happen. If it can be expressed in code, then that is better. The complete code can serve as a unit test.
```python
assert triangle.shape == (1, 1, 10, 10)
```

**Desktop (please complete the following information):**
 - Numpy Version [e.g. 1.22.0]
 - Pandas Version [e.g. 1.4.1]
 - Chainladder Version [e.g. 0.8.13]
