# Project Governance

This document defines the governance structure, contribution process, code ownership policies, and release lifecycle for the chainladder-python project under the Casualty Actuarial Society (CAS) Open-Source Projects Working Group (OSPWG).

---

## 1. Project Governance

### Organizational Structure

The project operates under the direction of the CAS Research Council, which identifies and oversees actuarial research projects, drives strategic industry innovations, and secures resources to address those emerging needs. The OSPWG appoints or elects project leadership to oversee the long-term direction of the project, supported by CAS staff and guided by the CAS Executive Council.

### Roles & Responsibilities

#### CAS Staff

CAS-appointed personnel oversee administrative logistics, continuity planning, resource allocation, and organizational support.

#### Working Group Leadership

The Working Group consists of an elected or appointed Chair and Vice Chair.

Following standard CAS volunteer governance, the Chair shall serve a three-year term, after which the Vice Chair rotates into the Chair position. As the Vice Chair rotates into the Chair position, the new Chairperson shall search and appoint a new Vice Chair or hold an election within the working group to appoint a new Vice Chair.

#### Project Maintainers (Codeowners)

Project Maintainers are trusted contributors with deep knowledge of the project.

Responsibilities include:

- Reviewing and approving pull requests (PRs)
- Triaging issues
- Managing the development backlog
- Maintaining code quality and project direction
- Participating in technical decision-making
- Working with the working group Chair and Vice Chair to coordinate outside communication among the CAS Research pillar, CAS Leadership Team, and with the CAS Executive Council

#### CAS GitHub Repo Members (Collaborators)

Membership in the CAS GitHub repo is open to community members who wish to publicly participate in project development.

Organization membership does not grant repository write on protected branches or administrative permissions. However, one will be able to create a branch on the repo instead of working off your own fork. You will also have elevated write privileges such as managing tickets and becoming a reviewer, and is the required pathway to become a Codeowner. For those who are interested in making consistent contributions, you are encouraged to join as a member.

#### Community Contributors

Community contributors include anyone submitting issues, documentation improvements, bug reports, PRs, or feature proposals. Everyone is welcome to join us and participate and collaborate.

---

## 2. Admin Rights and Decision Making

### Maintainers

A contributor becomes eligible for appointment as a Project Maintainer after successfully merging approximately 25 PRs into the main branch (1 weekly PR for about 6 months, and this is only a baseline for guidance). Once a contributor has successfully met this criteria, a Project Maintainer may invite them to become a codeowner depending on the overall level of contribution to the project. This initiation shall be reviewed and approved by all Project Maintainers who wish to voice their opinion.

Project Maintainers are expected to remain active participants in project development and review activities.

An active maintainer is defined as someone who has at least monthly activities on the repo, which may include reviewing PRs, managing issue tickets, making commits, performing maintenance work, or even doing activity outside of GitHub that involves the project such as promoting the project.

Active maintainer(s) may consider removing an inactive maintainer due to extended inactivity, resignation, or misconduct as needed. Maintainers may also step down themselves if they can no longer hold the responsibilities required.

- The project should maintain at least three active Project Maintainers.
- If fewer than three maintainers remain active, recruitment efforts should begin.
- If only one active maintainer remains, CAS staff will evaluate whether to recruit additional maintainers, temporarily assume administrative support, or pause project development.

### Technical Decision Making

Community contributors are encouraged to discuss review feedback and revise their work accordingly. However, project maintainers have final authority over technical decisions.

If maintainers themselves disagree on a significant technical decision, they should work toward consensus.

**Single-issue voting:**

- **+1**: approve.
- **0**: abstain. With a provided supporting reason.
- **-1**: disapprove. With a provided supporting reason.

**Choosing among a list of alternatives:**

- Ranked-choice voting

If consensus cannot be reached, additional maintainers should participate to resolve the disagreement. Maintainers may also seek comments from the working group or the general public.

### Governance Changes

Changes to this governance document require sharing proposed changes to the OSPWG for comments, but Project Maintainers shall have sole discretion on approving, rejecting, or revising any of the proposed changes based on the feedback received.

---

## 3. Development Standards

### Architecture

To maintain consistency with the broader Python data science ecosystem, the project follows established industry conventions.

#### Triangle & Options APIs

Interfaces should closely align with the Pandas DataFrame API whenever appropriate.

#### Estimator APIs

Estimator implementations should follow the Scikit-Learn estimator and transformer architecture.

### Coding Documentation Style

Code formatting and linting are enforced using Ruff.

Public parameter names should remain intuitive and consistent with Pandas and Scikit-Learn conventions whenever practical.

Documentation should:

- Follow NumPy docstring conventions
- Include executable examples where appropriate
- Remain concise and clear
- Be updated whenever user-facing behavior changes

---

## 4. Repository and Contribution

### Milestones & Goals

Project milestones and strategic goals are derived directly from community issue tickets on GitHub. Maintainers shall triage incoming tickets and tag selected items into targeted milestones to shape upcoming release cycles. To preserve transparency and keep all context in one place, all project-related communications shall remain on GitHub and avoid private discussion channels like chats or emails.

### Bug Reports and Enhancement Requests

All bug reports and feature requests must be opened as GitHub issues.

### Branching Strategy

The repository maintains three primary branches and other development branches.

| Branch | Description |
| --- | --- |
| `main` | Primary development branch. Must remain stable and pass all automated checks. Protected. |
| `release` | Contains the latest stable release published to PyPI and conda-forge. Recommended for production use. Protected. |
| `pre-release` | Houses code staged for an upcoming release. Unprotected. |
| `dev branches` | House feature development to be merged into main. Unprotected. |

Branches shall begin with # and the issue number, following by a short description of what the branch is (e.g. #123-new_friedland_dataset). Once the branch is merged into main, the owner shall consider deleting the branch that is no longer useful.

### Finding and Issue to Contribute to

You may claim or request assignment for any open issue, including inactive ones with existing assignees. Look for the “Great First Contribution!” label for beginner-friendly tasks.

### Creating a Development Environment

Please review our [guide](contributing.md) on how to set up your development environment and recommended workflows.

### AI Usage Policy

AI-assisted development is permitted. However, contributors using AI or LLMs must disclose its use in the PR description and verify that all code was manually reviewed and tested locally. The description can be brief, provided it gives the reviewer sufficient context on the tools used. Do not submit AI-generated "slop" - your time saving is not worth the increased burden on reviewers.

AI tools are powerful development tools to support your work but can still hallucinate or make errors if left unsupervised. As we work to round out our testing suite, we are still relying on human reviewers to ensure integrity of our codebase. So we would rather interact with you, not your tools. All external contributions and project interactions must be initiated by a human. (We don’t forbid bots; but if you want to hook up a bot to, say, deal with all the ruff violations, please announce via an issue or a PR.) We reserve the right to close any PR or low effort contribution identified as automated; if yours is closed in error, please leave a comment explaining your context to confirm you are human before resubmitting.

Please also review the CAS AI Policy for Research & Publications and make sure your contributions fully comply with its guidelines.

**Examples of acceptable AI usage:**

- Look up a feature of the programming language being used;
- Check a code snippet you produced for syntactical correctness;
- A second pair of eyes to double-check any manually-entered data or figures;
- As a sanity check/sounding board for any coding/design strategy;
- To check written work for possible typos/errors.

**Examples of unacceptable AI usage:**

- Using AI to produce code for the repository without fully reviewing and understanding the code;
- Asking AI to answer questions that should be directed to the project maintainers/other members;
- Failing to adequately document/disclose AI usage in PRs

### Pull Request Requirements & Checklist

**Every PR must:**

- Declare you are adhering to the standards outlined in the project Governing Doc
- Pass all required automated checks
- Have proper PR subject title that summarizes the changes, with a proper prefix:
    - `[BRK]` for breaking changes and deprecations and removals
    - `[CHORE]` for chores and maintenance tasks
    - `[DOCS]` for documentation
    - `[FEAT]` for enhancements
    - `[FIX]` for bug fixes
    - `[TST]` for unit testing
- Link to the associated issue(s), where applicable
- Declare that you are a human and not a bot, and this PR is not AI generated
- Declare and briefly describe any AI/LLM assistance used and the extent of usage during development
- Be independently reviewed before merging

**PRs should:**

- Remain small and focused
- Include appropriate unit tests
- Include proper type hinting
- Update documentation as needed
- Update docstrings when public APIs change

### Continuous Integration

All required automated checks must pass before a PR is merged.

Maintainers may override a failing automated check only when they determine that the failure does not indicate a defect in the proposed changes. Some examples include:

- Known false positives from tooling (for example, Codecov reporting uncovered lines outside the scope of the PR).
- Checks that fail due to external infrastructure, transient service outages, or unrelated upstream issues.
- Test failures resulting from an intentional breaking change that has been reviewed and approved, where updating or replacing the affected tests is part of the accepted design.

When overriding a failed check, the approving maintainer should document the reason in the PR discussion so future contributors understand why the exception was granted.

### Emergency Hotfixes

In the event of a critical production bug, security issue, or broken release requiring immediate action, a Project Maintainer may bypass the standard review and CI requirements to merge an emergency hotfix.

- This emergency bypass may only be used once for a given hotfix. Any subsequent changes must follow the standard contribution and review process.
- A post-merge review by at least one other Project Maintainer shall be completed as soon as reasonably possible. Any deficiencies identified during the review shall be addressed in a follow-up PR.
- Only Project Maintainers may approve or perform an emergency bypass.

### Review Requirements & Checklist

**Reviewers must verify:**

- The implementation addresses the associated issue(s)
- The implementation is appropriate, maintainable, and follows `ARCHITECTURE.md`
- AI disclosures are complete and usage is appropriate
- PR subject title is appropriate
- Relevant issue(s) is linked
- Documentation and tests are appropriate
- Additional reviewers are not needed for secondary review

**Reviewers may also:**

- Suggest implementation improvements
- Recommend follow-up issues
- Identify opportunities for future enhancements

### Review Principles

Code reviews prioritize improving the project over achieving perfection. If a PR is a clear net improvement, maintainers should merge it and track remaining tasks in follow-up issues.

If a PR is too large, reviewers may ask the submitter to reduce its scope or break it into smaller PRs. For large or complex features, contributors are encouraged to submit and merge unit tests prior to implementing functional code whenever practical. See [Contributing](contributing.md#pull-requests-prs) for how to run the test suite locally.

All code changes require manual review prior to merging. Reviewers must understand the purpose and implementation of all substantive code changes before approval. Documentation must be written in clear, concise English.

### Reviewer Assignment

Incoming PRs are randomly assigned to a codeowner to distribute maintenance work fairly. Authors should allow at least one week for initial review before requesting reassignment or additional reviewers.

### Abandoned Pull Requests

If requested changes remain unaddressed for more than one week, reviewers will follow up with the contributor. If there is no meaningful progress after the reminder, the PR may be closed. Contributors may reopen or resubmit the work whenever they are ready to resume.

### Contributor Recognition

Contributors who make sustained and meaningful contributions may be added to the Core Contributors list, which may highlight contribution history, actuarial credentials, and periods of active service.

---

## 5. Releases & Lifecycle

### Release Schedule

The project aims to publish releases approximately once per quarter or after reaching significant development milestones.

Release shall be maintained by one of the maintainers.

Release notes should include the following sections:

- **Enhancements:** New user-facing functionality and improvements: new estimators, methods, attributes, and options; new platform or dependency-version support; new bundled sample datasets; and new or expanded test coverage, etc.
- **Bug Fixes:** Corrections to incorrect or unintended behavior in existing functionality, including numerical fixes, data corrections, and edge-case handling.
- **Deprecations & Removals:** APIs, parameters, or backends that are newly deprecated (still functional but scheduled for removal) or removed in this release.
- **Maintenance:** Internal changes that don't alter public behavior: refactors, type annotations, tooling and CI, code cleanup, and all documentation changes (docstrings, examples, and guides).
- **Dependencies:** Changes to required or optional dependencies: added or removed packages, minimum/maximum version bounds, pins, and automated dependency bumps.
- **Contributors:** New contributors whose first PR to the project was merged in this release and returning contributors that have made a contribution since last release shall be included.

### Versioning

The project should follow semantic versioning, given a version number MAJOR.MINOR.PATCH, increment the:

1. MAJOR version when you make incompatible API changes
2. MINOR version when you add functionality in a backward compatible manner
3. PATCH version when you make backward compatible bug fixes

### Publishing

Chainladder packages will be published on the following platforms:

- GitHub
- PyPI
- Conda

The process for doing this is accomplished by:

1. **Prepping “prerelease”:** Drafting the release on the “prerelease" branch. That work should include
    - Version bump in `pyproject.toml`
    - Deprecation and removal updates for the upcoming release
    - Release notes update on the doc page (`docs/library/releases.md`)
2. **Publish on GitHub:** When ready, package the GitHub release with a version tag (in the format of “v0.0.0”) targeting “prerelease”
3. **PyPi upload:** Package will automatically be released to PyPI
4. **Conda upload:** After the package appears on PyPI, conda-forge opens a version-bump PR on. Once that PR (auto) merges, conda-forge builds and publishes the conda package (this usually takes 6 - 12 hours after PyPI)
5. **“main” merge:** Open a PR and merge `prerelease` into `main` so the default branch matches the released code

### Release Announcements

We will do our best attempt to notify the public via the following channels when a new release is published:

- “What’s New in Chainladder x.x.x” page in the docs doc site and its changelog
- LinkedIn carousel deck with CAS branding by the maintainer(s)
- LinkedIn blog post with code demos and plots by the maintainer(s)
- CAS Weekly Bulletin by the CAS Staff Person(s)

### Python and Dependency Support

Python versions are supported until their EOL date. The core PyData and scientific dependencies will be supported following the spec-0 convention.

### Deprecation Policy

Deprecated functionality should remain available for at least:

- Three months, or
- One minor release,

whichever is longer.

Deprecations should emit clear warnings whenever practical.

---

## 6. Repository Operations

### Security

The project is provided "as is" under the terms of the Mozilla Public License 2.0 and does not guarantee the availability, security, or suitability of the software for any particular purpose.

- GitHub Actions workflows should reference actions by immutable commit SHA whenever practical to reduce supply chain risk.
- Automated security tooling, such as CodeQL and dependency vulnerability scanning, may be used to identify potential issues during development. Maintainers may address reported vulnerabilities as time and volunteer availability permit.

---

## 7. Citation

If you use chainladder-python for a project or for a research project, we would appreciate appropriate citation of the package in any published work. The citation information is set out below in BibTeX format:

(please replace the “month”, “year”, and “version” as needed)

```bibtex
@software{chainladder,
 author       = {{The chainladder-python development team}},
 title        = {casact/chainladder-python: Property and Casualty Loss Reserving in Python},
 month        = jan,
 year         = 2026,
 version      = {v0.0.1},
 url          = {https://github.com/casact/chainladder-python},
}
```

---

## 8. Code of Conduct

As contributors and maintainers of chainladder-python, we are committed to fostering an open, welcoming, and respectful community.

Our guiding principle is: **Be excellent to each other, and drive the actuarial frontier forward!**

**Expected behavior:**

- Treat all contributors with respect and professionalism.
- Focus discussions on improving the project, not criticizing individuals.
- Provide constructive feedback and assume good intentions.
- Welcome contributors of all experience levels and backgrounds.

**Unacceptable behavior:**

- Harassment, discrimination, or personal attacks.
- Trolling, insulting, or deliberately disruptive comments.
- Sharing private information without permission.
- Over-relying on AI or automated systems in a way that shifts the burden of validation, debugging, or review onto other contributors.
- Try to sell anything such as work or service.
- Any behavior that creates a hostile project environment.

Project maintainers may remove comments, reject contributions, restrict participation, or take other actions when behavior does not align with this Code of Conduct. This Code of Conduct applies to all project spaces, including GitHub repositories, discussions, issues, PRs, and public spaces where contributors represent the project. Violations may be reported privately to the Project Maintainers or the CAS.
