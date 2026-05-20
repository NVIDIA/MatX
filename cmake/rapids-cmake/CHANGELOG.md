# rapids-cmake 26.04.00 (8 Apr 2026)

### 🚨 Breaking Changes
* Update CCCL to 3.3.0 pre-release by @bdice in https://github.com/rapidsai/rapids-cmake/pull/981
### 🐛 Bug Fixes
* FINAL_CODEBLOCK can see targets that should be global by @robertmaynard in https://github.com/rapidsai/rapids-cmake/pull/993
### 🚀 New Features
* Add URL option to fetch packages from GitHub tarballs by @bdice in https://github.com/rapidsai/rapids-cmake/pull/968
* CMake 3.30+ means `CMAKE_PROJECT_INCLUDE` can be a list by @robertmaynard in https://github.com/rapidsai/rapids-cmake/pull/984
* Update cuco version to fetch the cuda fancy iterators by @PointKernel in https://github.com/rapidsai/rapids-cmake/pull/987
### 🛠️ Improvements
* Forward-merge release/26.02 into main by @bdice in https://github.com/rapidsai/rapids-cmake/pull/974
* Drop Python 3.10 support by @gforsyth in https://github.com/rapidsai/rapids-cmake/pull/975
* Use verify-hardcoded-version pre-commit hook by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/977
* Don't hard-code branch names by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/978
* Update CCCL to 3.2.1 pre-release by @bdice in https://github.com/rapidsai/rapids-cmake/pull/979
* Add support for Python 3.14 by @gforsyth in https://github.com/rapidsai/rapids-cmake/pull/982


**Full Changelog**: https://github.com/rapidsai/rapids-cmake/compare/v26.04.00a...release/26.04

# rapids-cmake 26.02.00 (4 Feb 2026)

### 🚨 Breaking Changes
* Default to static linking of libcudart by @bdice in https://github.com/rapidsai/rapids-cmake/pull/949
* Update to CCCL v3.2.0 by @bdice in https://github.com/rapidsai/rapids-cmake/pull/955
### 🐛 Bug Fixes
* Update to a newer CCCL 3.2.x prerelease with a fix for environment queries by @bdice in https://github.com/rapidsai/rapids-cmake/pull/967
### 🚀 New Features
* Add support for finding CMake prefixes exported by Python packages by @vyasr in https://github.com/rapidsai/rapids-cmake/pull/951
### 🛠️ Improvements
* Use strict priority in CI conda tests by @bdice in https://github.com/rapidsai/rapids-cmake/pull/942
* Use strict priority in CI conda tests by @bdice in https://github.com/rapidsai/rapids-cmake/pull/947
* Upgrade to nvCOMP 5.1.0.21 by @bdice in https://github.com/rapidsai/rapids-cmake/pull/946
* Enable merge barriers by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/948
* feat(SABI): use stable abi naming when requested by @gforsyth in https://github.com/rapidsai/rapids-cmake/pull/954
* Bump cuco version to fetch HLL fix by @PointKernel in https://github.com/rapidsai/rapids-cmake/pull/952
* Clarify docs for `rapids_cuda_init_architectures()` by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/956
* Add CUDA 13.1 support by @bdice in https://github.com/rapidsai/rapids-cmake/pull/953
* Bump cuco version to fetch new HLL add_if API and bug fixes by @PointKernel in https://github.com/rapidsai/rapids-cmake/pull/958
* build and test against CUDA 13.1.0 by @jameslamb in https://github.com/rapidsai/rapids-cmake/pull/960
* Update to a newer CCCL 3.2.x prerelease with a fix for mdspan by @bdice in https://github.com/rapidsai/rapids-cmake/pull/959
* Update to a newer CCCL 3.2.x prelease  by @nirandaperera in https://github.com/rapidsai/rapids-cmake/pull/961
* Use main shared-workflows branch by @jameslamb in https://github.com/rapidsai/rapids-cmake/pull/962
* Cherry-pick 25.12 changelog to main by @AyodeAwe in https://github.com/rapidsai/rapids-cmake/pull/963
* Fix update-version.sh to not modify cmake_minimum_required by @AyodeAwe in https://github.com/rapidsai/rapids-cmake/pull/964
* fix(build): build package on merge to `release/*` branch by @gforsyth in https://github.com/rapidsai/rapids-cmake/pull/971
* Update CCCL to final v3.2.0 release by @bdice in https://github.com/rapidsai/rapids-cmake/pull/970


**Full Changelog**: https://github.com/rapidsai/rapids-cmake/compare/v26.02.00a...release/26.02

# rapids-cmake 25.12.00 (10 Dec 2025)

### 🚨 Breaking Changes
* Remove deprecated support for fmt and spdlog by @bdice in https://github.com/rapidsai/rapids-cmake/pull/914
* Remove deprecated support for CUDA 11 by @bdice in https://github.com/rapidsai/rapids-cmake/pull/913
* Remove spdlog and fmt and move to rapids-logger 0.2.1 by @vyasr in https://github.com/rapidsai/rapids-cmake/pull/918
* Require CUDA 12.2+ by @jakirkham in https://github.com/rapidsai/rapids-cmake/pull/934
### 🐛 Bug Fixes
* Fix `RAPIDS_BRANCH` file by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/910
* Temporarily restore spdlog and fmt by @vyasr in https://github.com/rapidsai/rapids-cmake/pull/917
* Update rapids-logger to 0.2.2 by @vyasr in https://github.com/rapidsai/rapids-cmake/pull/920
* Can't use Genex in `GLOBAL_TARGETS` by @robertmaynard in https://github.com/rapidsai/rapids-cmake/pull/921
* Update rapids-logger to 0.2.3 by @vyasr in https://github.com/rapidsai/rapids-cmake/pull/923
* refactored update-version.sh to handle new branching strategy by @rockhowse in https://github.com/rapidsai/rapids-cmake/pull/936
### 🚀 New Features
* Adding `ENABLE_UNSTABLE` option for `rapids_cpm_cccl` by @nirandaperera in https://github.com/rapidsai/rapids-cmake/pull/919
### 🛠️ Improvements
* Update nvbench by @bdice in https://github.com/rapidsai/rapids-cmake/pull/895
* Update to CCCL v3.1.0 by @bdice in https://github.com/rapidsai/rapids-cmake/pull/906
* Upgrade to CPM 0.42.0 by @vyasr in https://github.com/rapidsai/rapids-cmake/pull/916
* Forward-merge branch-25.10 into branch-25.12 by @bdice in https://github.com/rapidsai/rapids-cmake/pull/927
* Update CCCL to final v3.1.0 release tag by @bdice in https://github.com/rapidsai/rapids-cmake/pull/928
* Use main in RAPIDS_BRANCH by @bdice in https://github.com/rapidsai/rapids-cmake/pull/930
* Use main branch by @bdice in https://github.com/rapidsai/rapids-cmake/pull/931
* Use main shared-workflows branch by @bdice in https://github.com/rapidsai/rapids-cmake/pull/932
* Use SPDX for all copyright headers by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/933
* Bump cuco version to fetch stream-ordered allocator by @PointKernel in https://github.com/rapidsai/rapids-cmake/pull/929
* Bump cuco version to bypass equal check for multiset/multimap by @PointKernel in https://github.com/rapidsai/rapids-cmake/pull/935
* Some more SPDX fixes by @KyleFromNVIDIA in https://github.com/rapidsai/rapids-cmake/pull/937
* Update CCCL to get backports for XGBoost compatibility by @bdice in https://github.com/rapidsai/rapids-cmake/pull/944

## New Contributors
* @nirandaperera made their first contribution in https://github.com/rapidsai/rapids-cmake/pull/919
* @rockhowse made their first contribution in https://github.com/rapidsai/rapids-cmake/pull/936

**Full Changelog**: https://github.com/rapidsai/rapids-cmake/compare/v25.12.00a...release/25.12

# rapids-cmake 25.10.00 (8 Oct 2025)

## 🚨 Breaking Changes

- Upgrade to nvCOMP 5.0.0.6 ([#896](https://github.com/rapidsai/rapids-cmake/pull/896)) [@vuule](https://github.com/vuule)

## 🐛 Bug Fixes

- Fix generation of `RAPIDS_BRANCH` ([#909](https://github.com/rapidsai/rapids-cmake/pull/909)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- change Architectures to avoid compute_100/120 code gen bug ([#900](https://github.com/rapidsai/rapids-cmake/pull/900)) [@robertmaynard](https://github.com/robertmaynard)
- set_architectures + NATIVE works even without a GPU ([#899](https://github.com/rapidsai/rapids-cmake/pull/899)) [@robertmaynard](https://github.com/robertmaynard)
- set_architectures correctly limits CUDA 12.8 to sm_120 ([#898](https://github.com/rapidsai/rapids-cmake/pull/898)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_export handles PROJECT_VERSION CMake 4.1+ change in behavior ([#893](https://github.com/rapidsai/rapids-cmake/pull/893)) [@robertmaynard](https://github.com/robertmaynard)
- Fix detect architectures to support gcc 14 ([#892](https://github.com/rapidsai/rapids-cmake/pull/892)) [@ChuckHastings](https://github.com/ChuckHastings)
- Update RAPIDS.cmake to match new logic in other projects ([#884](https://github.com/rapidsai/rapids-cmake/pull/884)) [@robertmaynard](https://github.com/robertmaynard)
- Update RAPIDS_BRANCH to 25.10 ([#883](https://github.com/rapidsai/rapids-cmake/pull/883)) [@robertmaynard](https://github.com/robertmaynard)
- Expand versions.json to have src_dir representation ([#882](https://github.com/rapidsai/rapids-cmake/pull/882)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Update cuco version to fetch signed comparison warning fix ([#905](https://github.com/rapidsai/rapids-cmake/pull/905)) [@PointKernel](https://github.com/PointKernel)
- Updated dependencies.yaml to support CUDA 13 ([#901](https://github.com/rapidsai/rapids-cmake/pull/901)) [@robertmaynard](https://github.com/robertmaynard)
- Use family arch flags whenever possible ([#897](https://github.com/rapidsai/rapids-cmake/pull/897)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade to nvCOMP 5.0.0.6 ([#896](https://github.com/rapidsai/rapids-cmake/pull/896)) [@vuule](https://github.com/vuule)
- Update set_architectures to support CUDA 13 ([#891](https://github.com/rapidsai/rapids-cmake/pull/891)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake Expose `rapids_cuda_enable_fatbin_compression` to have better CUDA 12/13 compression API ([#890](https://github.com/rapidsai/rapids-cmake/pull/890)) [@robertmaynard](https://github.com/robertmaynard)
- CUDA 13 builds will properly pull nvcomp  4.2.0.11 ([#889](https://github.com/rapidsai/rapids-cmake/pull/889)) [@robertmaynard](https://github.com/robertmaynard)
- Properly store `source_subdir` values from `rapids_cpm_find` calls ([#885](https://github.com/rapidsai/rapids-cmake/pull/885)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Configure repo for automatic release notes generation ([#907](https://github.com/rapidsai/rapids-cmake/pull/907)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use branch-25.10 again ([#904](https://github.com/rapidsai/rapids-cmake/pull/904)) [@jameslamb](https://github.com/jameslamb)
- Update rapids-dependency-file-generator ([#903](https://github.com/rapidsai/rapids-cmake/pull/903)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update cuco version to fetch the new roaring bitmap ([#902](https://github.com/rapidsai/rapids-cmake/pull/902)) [@PointKernel](https://github.com/PointKernel)
- Bump cuco version to fetch a bug fix for device retrieve APIs ([#887](https://github.com/rapidsai/rapids-cmake/pull/887)) [@PointKernel](https://github.com/PointKernel)
- chore: update docker version tags ([#873](https://github.com/rapidsai/rapids-cmake/pull/873)) [@gforsyth](https://github.com/gforsyth)

# rapids-cmake 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- rapids_cpm_cccl: Remove support for CCCL &lt; 2.8 ([#859](https://github.com/rapidsai/rapids-cmake/pull/859)) [@robertmaynard](https://github.com/robertmaynard)
- Remove CUDA 11 support ([#855](https://github.com/rapidsai/rapids-cmake/pull/855)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update to CCCL 3.0 ([#854](https://github.com/rapidsai/rapids-cmake/pull/854)) [@vyasr](https://github.com/vyasr)
- Require cpp subdirectory for RMM ([#832](https://github.com/rapidsai/rapids-cmake/pull/832)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- CCCL: disable PDL ([#876](https://github.com/rapidsai/rapids-cmake/pull/876)) [@bdice](https://github.com/bdice)
- Use RMM main (new branching strategy) to fix downstream fetching issues ([#862](https://github.com/rapidsai/rapids-cmake/pull/862)) [@bdice](https://github.com/bdice)
- rapids_cpm_cccl: Update to new location of cccl-config ([#858](https://github.com/rapidsai/rapids-cmake/pull/858)) [@robertmaynard](https://github.com/robertmaynard)
- Remove CCCL patches that aren&#39;t used anymore ([#857](https://github.com/rapidsai/rapids-cmake/pull/857)) [@robertmaynard](https://github.com/robertmaynard)
- Fetch the atomic fix in CCCL 3.0 ([#856](https://github.com/rapidsai/rapids-cmake/pull/856)) [@PointKernel](https://github.com/PointKernel)

## 📖 Documentation

- add docs on CI workflow inputs ([#868](https://github.com/rapidsai/rapids-cmake/pull/868)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Update CCCL version tag for PDL disable ([#879](https://github.com/rapidsai/rapids-cmake/pull/879)) [@davidwendt](https://github.com/davidwendt)
- Use `RAPIDS_BRANCH` file to handle the new branching strategy ([#870](https://github.com/rapidsai/rapids-cmake/pull/870)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake: Add support for a version suffix to mean using main ([#864](https://github.com/rapidsai/rapids-cmake/pull/864)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cpm_cccl: Remove support for CCCL &lt; 2.8 ([#859](https://github.com/rapidsai/rapids-cmake/pull/859)) [@robertmaynard](https://github.com/robertmaynard)
- Update to CCCL 3.0 ([#831](https://github.com/rapidsai/rapids-cmake/pull/831)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Update to CCCL v3.0.2 ([#878](https://github.com/rapidsai/rapids-cmake/pull/878)) [@bdice](https://github.com/bdice)
- fix(docker): use versioned `-latest` tag for all `rapidsai` images ([#871](https://github.com/rapidsai/rapids-cmake/pull/871)) [@gforsyth](https://github.com/gforsyth)
- Revert &quot;Use RMM main (new branching strategy)&quot; ([#869](https://github.com/rapidsai/rapids-cmake/pull/869)) [@robertmaynard](https://github.com/robertmaynard)
- Rename `*.hpp.in` to `*.h.in` to signify that they are C headers ([#867](https://github.com/rapidsai/rapids-cmake/pull/867)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor(shellcheck): enable for all files ([#866](https://github.com/rapidsai/rapids-cmake/pull/866)) [@gforsyth](https://github.com/gforsyth)
- Remove nvidia and dask channels ([#865](https://github.com/rapidsai/rapids-cmake/pull/865)) [@vyasr](https://github.com/vyasr)
- Upgrade cuCollections to fetch the new storage for better runtime performance ([#861](https://github.com/rapidsai/rapids-cmake/pull/861)) [@PointKernel](https://github.com/PointKernel)
- Remove CUDA 11 support ([#855](https://github.com/rapidsai/rapids-cmake/pull/855)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update to CCCL 3.0 ([#854](https://github.com/rapidsai/rapids-cmake/pull/854)) [@vyasr](https://github.com/vyasr)
- Deprecate fmt and spdlog ([#853](https://github.com/rapidsai/rapids-cmake/pull/853)) [@vyasr](https://github.com/vyasr)
- Forward-merge branch-25.06 into branch-25.08 ([#848](https://github.com/rapidsai/rapids-cmake/pull/848)) [@gforsyth](https://github.com/gforsyth)
- Update to NVTX 3.2.0. ([#844](https://github.com/rapidsai/rapids-cmake/pull/844)) [@bdice](https://github.com/bdice)
- Forward-merge branch-25.06 into branch-25.08 ([#839](https://github.com/rapidsai/rapids-cmake/pull/839)) [@gforsyth](https://github.com/gforsyth)
- Temporarily use patched CCCL ([#833](https://github.com/rapidsai/rapids-cmake/pull/833)) [@bdice](https://github.com/bdice)
- Require cpp subdirectory for RMM ([#832](https://github.com/rapidsai/rapids-cmake/pull/832)) [@bdice](https://github.com/bdice)

# rapids-cmake 25.06.00 (5 Jun 2025)

## 🚨 Breaking Changes

- Make `RAPIDS.cmake` inlineable without requiring updates ([#825](https://github.com/rapidsai/rapids-cmake/pull/825)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Make `RAPIDS.cmake` inlineable without requiring updates ([#822](https://github.com/rapidsai/rapids-cmake/pull/822)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add build patch functionality ([#801](https://github.com/rapidsai/rapids-cmake/pull/801)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 🐛 Bug Fixes

- Get newer CCCL 2.8 with fix for atomics. ([#840](https://github.com/rapidsai/rapids-cmake/pull/840)) [@bdice](https://github.com/bdice)
- Unset CUDAARCHS in test ([#835](https://github.com/rapidsai/rapids-cmake/pull/835)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Revert &quot;Make `RAPIDS.cmake` inlineable without requiring updates&quot; ([#824](https://github.com/rapidsai/rapids-cmake/pull/824)) [@bdice](https://github.com/bdice)
- Add `include_guard(GLOBAL)` to download_with_retry.cmake ([#818](https://github.com/rapidsai/rapids-cmake/pull/818)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix issue with how project pins are determined ([#815](https://github.com/rapidsai/rapids-cmake/pull/815)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Disable git_shallow for nvtx3 and bs_thread_pool ([#811](https://github.com/rapidsai/rapids-cmake/pull/811)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix NVTX3 export ([#804](https://github.com/rapidsai/rapids-cmake/pull/804)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 🚀 New Features

- Upgrade nvbench for latest throttling control updates ([#830](https://github.com/rapidsai/rapids-cmake/pull/830)) [@PointKernel](https://github.com/PointKernel)
- Add build patch functionality ([#801](https://github.com/rapidsai/rapids-cmake/pull/801)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 🛠️ Improvements

- explicitly provide &#39;script&#39; to CI workflows ([#849](https://github.com/rapidsai/rapids-cmake/pull/849)) [@jameslamb](https://github.com/jameslamb)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#847](https://github.com/rapidsai/rapids-cmake/pull/847)) [@bdice](https://github.com/bdice)
- Update to clang 20 ([#846](https://github.com/rapidsai/rapids-cmake/pull/846)) [@bdice](https://github.com/bdice)
- Update to CCCL 2.8.4. ([#845](https://github.com/rapidsai/rapids-cmake/pull/845)) [@bdice](https://github.com/bdice)
- Bump cuco version to fetch multiple CCCL upgrade fixes ([#843](https://github.com/rapidsai/rapids-cmake/pull/843)) [@PointKernel](https://github.com/PointKernel)
- Build and test with CUDA 12.9.0 ([#842](https://github.com/rapidsai/rapids-cmake/pull/842)) [@bdice](https://github.com/bdice)
- Upgrade cuco version to include murmurhash performance fix ([#838](https://github.com/rapidsai/rapids-cmake/pull/838)) [@PointKernel](https://github.com/PointKernel)
- feat: add Python 3.13 support ([#834](https://github.com/rapidsai/rapids-cmake/pull/834)) [@gforsyth](https://github.com/gforsyth)
- Update cuCollections to avoid CCCL deprecation warnings ([#828](https://github.com/rapidsai/rapids-cmake/pull/828)) [@miscco](https://github.com/miscco)
- Use cpp subdirectory for RMM. ([#827](https://github.com/rapidsai/rapids-cmake/pull/827)) [@bdice](https://github.com/bdice)
- Make `RAPIDS.cmake` inlineable without requiring updates ([#825](https://github.com/rapidsai/rapids-cmake/pull/825)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove hard-coded RAPIDS versions ([#823](https://github.com/rapidsai/rapids-cmake/pull/823)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Make `RAPIDS.cmake` inlineable without requiring updates ([#822](https://github.com/rapidsai/rapids-cmake/pull/822)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Format all CMake code ([#821](https://github.com/rapidsai/rapids-cmake/pull/821)) [@bdice](https://github.com/bdice)
- In conda-build produced binaries, use paths relative to `$PREFIX` ([#817](https://github.com/rapidsai/rapids-cmake/pull/817)) [@jakirkham](https://github.com/jakirkham)
- Remove CMP0135 policy checks. ([#816](https://github.com/rapidsai/rapids-cmake/pull/816)) [@bdice](https://github.com/bdice)
- Use hashes instead of tags ([#812](https://github.com/rapidsai/rapids-cmake/pull/812)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor(conda): remove unused conda package and associated CI resources ([#810](https://github.com/rapidsai/rapids-cmake/pull/810)) [@gforsyth](https://github.com/gforsyth)
- Allow retrying downloads on network failures ([#809](https://github.com/rapidsai/rapids-cmake/pull/809)) [@pentschev](https://github.com/pentschev)
- Define `-DCUB_DISABLE_NAMESPACE_MAGIC` and `-DCUB_IGNORE_NAMESPACE_MAGIC_ERROR` in `rapids_cpm_cccl` ([#807](https://github.com/rapidsai/rapids-cmake/pull/807)) [@trxcllnt](https://github.com/trxcllnt)
- Update to CCCL 2.8.3 ([#793](https://github.com/rapidsai/rapids-cmake/pull/793)) [@bdice](https://github.com/bdice)

# rapids-cmake 25.04.00 (9 Apr 2025)

## 🚨 Breaking Changes

- Backport build patch functionality ([#805](https://github.com/rapidsai/rapids-cmake/pull/805)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update gtest version ([#795](https://github.com/rapidsai/rapids-cmake/pull/795)) [@vyasr](https://github.com/vyasr)
- Remove now superfluous fmt patch ([#790](https://github.com/rapidsai/rapids-cmake/pull/790)) [@vyasr](https://github.com/vyasr)
- Use CTest dynamic resource spec generation ([#786](https://github.com/rapidsai/rapids-cmake/pull/786)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Switch to using rapids-logger as a library ([#765](https://github.com/rapidsai/rapids-cmake/pull/765)) [@vyasr](https://github.com/vyasr)
- Require CMake 3.30.4, and latest CPM ([#753](https://github.com/rapidsai/rapids-cmake/pull/753)) [@robertmaynard](https://github.com/robertmaynard)

## 🐛 Bug Fixes

- Set correct compiler feature for `generate_ctest_json` ([#802](https://github.com/rapidsai/rapids-cmake/pull/802)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Also download fmt when building nvbench statically ([#792](https://github.com/rapidsai/rapids-cmake/pull/792)) [@vyasr](https://github.com/vyasr)
- Rollback to CPM 0.40.0 ([#789](https://github.com/rapidsai/rapids-cmake/pull/789)) [@robertmaynard](https://github.com/robertmaynard)
- Sanitize patch inputs so that quotes are properly escaped ([#787](https://github.com/rapidsai/rapids-cmake/pull/787)) [@robertmaynard](https://github.com/robertmaynard)
- Fix tests broken by rapids-logger changes ([#775](https://github.com/rapidsai/rapids-cmake/pull/775)) [@vyasr](https://github.com/vyasr)

## 🚀 New Features

- Backport build patch functionality ([#805](https://github.com/rapidsai/rapids-cmake/pull/805)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add support for components in Cython `rapids_cython_create_modules` ([#783](https://github.com/rapidsai/rapids-cmake/pull/783)) [@pentschev](https://github.com/pentschev)
- rapids_cpm_package_override: Always load overrides ([#778](https://github.com/rapidsai/rapids-cmake/pull/778)) [@robertmaynard](https://github.com/robertmaynard)
- Require CMake 3.30.4, and latest CPM ([#753](https://github.com/rapidsai/rapids-cmake/pull/753)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Update gtest version ([#795](https://github.com/rapidsai/rapids-cmake/pull/795)) [@vyasr](https://github.com/vyasr)
- Remove now superfluous fmt patch ([#790](https://github.com/rapidsai/rapids-cmake/pull/790)) [@vyasr](https://github.com/vyasr)
- Use CTest dynamic resource spec generation ([#786](https://github.com/rapidsai/rapids-cmake/pull/786)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Use conda-build instead of conda-mambabuild ([#784](https://github.com/rapidsai/rapids-cmake/pull/784)) [@bdice](https://github.com/bdice)
- Set `-Wno-deprecated-gpu-targets` when compiling for RAPIDS architectures ([#776](https://github.com/rapidsai/rapids-cmake/pull/776)) [@bdice](https://github.com/bdice)
- Add shellcheck to pre-commit and fix warnings ([#772](https://github.com/rapidsai/rapids-cmake/pull/772)) [@gforsyth](https://github.com/gforsyth)
- Update `cuCollection` to not rely on `thrust::identity` ([#771](https://github.com/rapidsai/rapids-cmake/pull/771)) [@miscco](https://github.com/miscco)
- Use rapids-generate-version for package versions ([#770](https://github.com/rapidsai/rapids-cmake/pull/770)) [@bdice](https://github.com/bdice)
- Use shared-workflows branch-25.04 ([#768](https://github.com/rapidsai/rapids-cmake/pull/768)) [@bdice](https://github.com/bdice)
- add telemetry ([#767](https://github.com/rapidsai/rapids-cmake/pull/767)) [@msarahan](https://github.com/msarahan)
- Add build_type input field for `test.yaml` ([#766](https://github.com/rapidsai/rapids-cmake/pull/766)) [@gforsyth](https://github.com/gforsyth)
- Switch to using rapids-logger as a library ([#765](https://github.com/rapidsai/rapids-cmake/pull/765)) [@vyasr](https://github.com/vyasr)
- Add `verify-codeowners` hook ([#762](https://github.com/rapidsai/rapids-cmake/pull/762)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Forward-merge branch-25.02 to branch-25.04 ([#760](https://github.com/rapidsai/rapids-cmake/pull/760)) [@bdice](https://github.com/bdice)
- Migrate to NVKS for amd64 CI runners ([#758](https://github.com/rapidsai/rapids-cmake/pull/758)) [@bdice](https://github.com/bdice)

# rapids-cmake 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- Update to CCCL 2.7.0-rc2. ([#710](https://github.com/rapidsai/rapids-cmake/pull/710)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- Sanitize patch input to protect against CMake Language limitations ([#773](https://github.com/rapidsai/rapids-cmake/pull/773)) [@robertmaynard](https://github.com/robertmaynard)
- Backport cccl PR 3578 to v2.7 to unblock CUDA 12.8 builds ([#756](https://github.com/rapidsai/rapids-cmake/pull/756)) [@robertmaynard](https://github.com/robertmaynard)
- Use sysroot 2.17 on CUDA &lt; 11.8. ([#745](https://github.com/rapidsai/rapids-cmake/pull/745)) [@bdice](https://github.com/bdice)
- rapids_cpm_cccl use proper CCCL version value to compute install rules ([#742](https://github.com/rapidsai/rapids-cmake/pull/742)) [@robertmaynard](https://github.com/robertmaynard)
- Correct nvbench + conda test to explicitly pull spdlog ([#733](https://github.com/rapidsai/rapids-cmake/pull/733)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- fix signature in rapids_export() docs ([#732](https://github.com/rapidsai/rapids-cmake/pull/732)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Add support for new architectures in CUDA 12.8 ([#746](https://github.com/rapidsai/rapids-cmake/pull/746)) [@robertmaynard](https://github.com/robertmaynard)
- Update logger hash ([#738](https://github.com/rapidsai/rapids-cmake/pull/738)) [@vyasr](https://github.com/vyasr)
- Add rapids-logger ([#737](https://github.com/rapidsai/rapids-cmake/pull/737)) [@vyasr](https://github.com/vyasr)
- Fea/support embedded patch content ([#728](https://github.com/rapidsai/rapids-cmake/pull/728)) [@robertmaynard](https://github.com/robertmaynard)
- Update to CCCL 2.7.0-rc2. ([#710](https://github.com/rapidsai/rapids-cmake/pull/710)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Revert CUDA 12.8 shared workflow branch changes ([#759](https://github.com/rapidsai/rapids-cmake/pull/759)) [@vyasr](https://github.com/vyasr)
- Build and test with CUDA 12.8.0 ([#754](https://github.com/rapidsai/rapids-cmake/pull/754)) [@bdice](https://github.com/bdice)
- Allow sysroot 2.28 in CUDA 11 jobs. ([#749](https://github.com/rapidsai/rapids-cmake/pull/749)) [@bdice](https://github.com/bdice)
- Fetch cuco filter alignment fix to unblock RAPIDS CI ([#744](https://github.com/rapidsai/rapids-cmake/pull/744)) [@PointKernel](https://github.com/PointKernel)
- Use CCCL v2.7.0 final tag. ([#743](https://github.com/rapidsai/rapids-cmake/pull/743)) [@bdice](https://github.com/bdice)
- Use GCC 13 in CUDA 12 conda builds. ([#741](https://github.com/rapidsai/rapids-cmake/pull/741)) [@bdice](https://github.com/bdice)
- Add new global targets for rmm ([#739](https://github.com/rapidsai/rapids-cmake/pull/739)) [@vyasr](https://github.com/vyasr)
- remove setup.cfg, use pyproject.toml ([#736](https://github.com/rapidsai/rapids-cmake/pull/736)) [@jameslamb](https://github.com/jameslamb)
- Bump cuco by one commit to include `cuco::arrow_filter_policy` updates ([#735](https://github.com/rapidsai/rapids-cmake/pull/735)) [@mhaseeb123](https://github.com/mhaseeb123)
- Require Cython to be found ([#734](https://github.com/rapidsai/rapids-cmake/pull/734)) [@vyasr](https://github.com/vyasr)
- Update version references in workflow ([#729](https://github.com/rapidsai/rapids-cmake/pull/729)) [@AyodeAwe](https://github.com/AyodeAwe)
- Bump cuco version to fetch various fixes and new features ([#723](https://github.com/rapidsai/rapids-cmake/pull/723)) [@PointKernel](https://github.com/PointKernel)
- Add breaking change workflow trigger ([#719](https://github.com/rapidsai/rapids-cmake/pull/719)) [@AyodeAwe](https://github.com/AyodeAwe)

# rapids-cmake 24.12.00 (11 Dec 2024)

## 🐛 Bug Fixes

- Fix failures in cpm_generate_pins-nested tests ([#724](https://github.com/rapidsai/rapids-cmake/pull/724)) [@vyasr](https://github.com/vyasr)
- Search for the normalized package name when patching ([#715](https://github.com/rapidsai/rapids-cmake/pull/715)) [@vyasr](https://github.com/vyasr)
- rapids_export_write_language no longer generates cmake dev warnings ([#712](https://github.com/rapidsai/rapids-cmake/pull/712)) [@robertmaynard](https://github.com/robertmaynard)
- Update rapids-cmake to support cccl 2.8 new install rules ([#711](https://github.com/rapidsai/rapids-cmake/pull/711)) [@robertmaynard](https://github.com/robertmaynard)
- gtest_discover_tests doesn&#39;t crash rapids_test_install_relocatable ([#697](https://github.com/rapidsai/rapids-cmake/pull/697)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Fix docs warning about get_html_theme_path. ([#705](https://github.com/rapidsai/rapids-cmake/pull/705)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Verifies that current_json_dir placeholder works with normalized package names ([#716](https://github.com/rapidsai/rapids-cmake/pull/716)) [@robertmaynard](https://github.com/robertmaynard)
- Allow for cpm override via file and direct calls to rapids_cpm_package_override ([#714](https://github.com/rapidsai/rapids-cmake/pull/714)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade nvcomp to 4.1.0.6 ([#709](https://github.com/rapidsai/rapids-cmake/pull/709)) [@bdice](https://github.com/bdice)
- Warn when override of a default package uses the incorrect case ([#708](https://github.com/rapidsai/rapids-cmake/pull/708)) [@robertmaynard](https://github.com/robertmaynard)
- Allow CCCL 2.7 to be used in a rapids-cmake override ([#706](https://github.com/rapidsai/rapids-cmake/pull/706)) [@robertmaynard](https://github.com/robertmaynard)
- version.json patch command supports required field ([#700](https://github.com/rapidsai/rapids-cmake/pull/700)) [@robertmaynard](https://github.com/robertmaynard)
- Bump cuco version to fetch the Hyperloglog renaming ([#698](https://github.com/rapidsai/rapids-cmake/pull/698)) [@PointKernel](https://github.com/PointKernel)

## 🛠️ Improvements

- Bump cuco version to fetch the new multiset retrieve ([#707](https://github.com/rapidsai/rapids-cmake/pull/707)) [@PointKernel](https://github.com/PointKernel)

# rapids-cmake 24.10.00 (9 Oct 2024)

## 🚨 Breaking Changes

- update fmt (to 11.0.2) and spdlog (to 1.14.1) ([#689](https://github.com/rapidsai/rapids-cmake/pull/689)) [@jameslamb](https://github.com/jameslamb)
- Remove 24.10 deprecated commands ([#665](https://github.com/rapidsai/rapids-cmake/pull/665)) [@robertmaynard](https://github.com/robertmaynard)

## 🐛 Bug Fixes

- Add workaround for gcc bug in cuco ([#685](https://github.com/rapidsai/rapids-cmake/pull/685)) [@miscco](https://github.com/miscco)
- Revert &quot;Update to CPM 0.40.2 to fix CMake 3.30 deprecation warnings ([#678)&quot; (#679](https://github.com/rapidsai/rapids-cmake/pull/678)&quot; (#679)) [@jameslamb](https://github.com/jameslamb)
- Update to CPM 0.40.2 to fix CMake 3.30 deprecation warnings ([#678](https://github.com/rapidsai/rapids-cmake/pull/678)) [@robertmaynard](https://github.com/robertmaynard)
- Bump cuco version to fetch several bug fixes ([#677](https://github.com/rapidsai/rapids-cmake/pull/677)) [@PointKernel](https://github.com/PointKernel)
- rapids_cpm_nvcomp has consistent behavior on every cmake execution ([#676](https://github.com/rapidsai/rapids-cmake/pull/676)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake generated C++ files have current copyright year ([#674](https://github.com/rapidsai/rapids-cmake/pull/674)) [@robertmaynard](https://github.com/robertmaynard)
- `write_language` now properly walks up the `add_subdirectory` call stack ([#671](https://github.com/rapidsai/rapids-cmake/pull/671)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Update docs for overriding rapids-cmake ([#681](https://github.com/rapidsai/rapids-cmake/pull/681)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- add ASSOCIATED_TARGETS to function signature in rapids_cython_create_modules() docs ([#670](https://github.com/rapidsai/rapids-cmake/pull/670)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Reduce cpm tests network usage ([#683](https://github.com/rapidsai/rapids-cmake/pull/683)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake tests no longer download CPM when `NO_CPM_CACHE` is set. ([#682](https://github.com/rapidsai/rapids-cmake/pull/682)) [@robertmaynard](https://github.com/robertmaynard)
- Remove deprecated rapids_export_find_package_* signatures ([#666](https://github.com/rapidsai/rapids-cmake/pull/666)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade nvcomp to 4.0.1 ([#633](https://github.com/rapidsai/rapids-cmake/pull/633)) [@vuule](https://github.com/vuule)

## 🛠️ Improvements

- Add INSTALL_TARGET argument to rapids_add_test() ([#692](https://github.com/rapidsai/rapids-cmake/pull/692)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#691](https://github.com/rapidsai/rapids-cmake/pull/691)) [@jameslamb](https://github.com/jameslamb)
- update fmt (to 11.0.2) and spdlog (to 1.14.1) ([#689](https://github.com/rapidsai/rapids-cmake/pull/689)) [@jameslamb](https://github.com/jameslamb)
- Add support for Python 3.12 ([#688](https://github.com/rapidsai/rapids-cmake/pull/688)) [@jameslamb](https://github.com/jameslamb)
- Update rapidsai/pre-commit-hooks ([#686](https://github.com/rapidsai/rapids-cmake/pull/686)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Drop Python 3.9 support ([#684](https://github.com/rapidsai/rapids-cmake/pull/684)) [@jameslamb](https://github.com/jameslamb)
- Allow nested lib location for nvcomp ([#675](https://github.com/rapidsai/rapids-cmake/pull/675)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update pre-commit hooks ([#669](https://github.com/rapidsai/rapids-cmake/pull/669)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#668](https://github.com/rapidsai/rapids-cmake/pull/668)) [@bdice](https://github.com/bdice)
- Remove 24.10 deprecated commands ([#665](https://github.com/rapidsai/rapids-cmake/pull/665)) [@robertmaynard](https://github.com/robertmaynard)

# rapids-cmake 24.08.00 (7 Aug 2024)

## 🚨 Breaking Changes

- Move required CMake version to 3.26.4 ([#627](https://github.com/rapidsai/rapids-cmake/pull/627)) [@robertmaynard](https://github.com/robertmaynard)
- Removes legacy rapids-cmake cython implementations as it is deprecated in 24.08 ([#614](https://github.com/rapidsai/rapids-cmake/pull/614)) [@robertmaynard](https://github.com/robertmaynard)
- Update CCCL to v2.5.0 ([#607](https://github.com/rapidsai/rapids-cmake/pull/607)) [@trxcllnt](https://github.com/trxcllnt)

## 🐛 Bug Fixes

- bs_thread_pool uses C++ 17 ([#662](https://github.com/rapidsai/rapids-cmake/pull/662)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Use `CMAKE_CUDA_ARCHITECTURES` value when ENV{CUDAARCHS} is set ([#659](https://github.com/rapidsai/rapids-cmake/pull/659)) [@robertmaynard](https://github.com/robertmaynard)
- Pass `GLOBAL_TARGETS` to `rapids_cpm_find()` in `rapids_cpm_bs_thread_pool()` ([#655](https://github.com/rapidsai/rapids-cmake/pull/655)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add rapids_cpm_nvtx3 to cmake-format-rapids-cmake.json ([#652](https://github.com/rapidsai/rapids-cmake/pull/652)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- generate_resource_spec uses the enabled languages to determine compiler ([#645](https://github.com/rapidsai/rapids-cmake/pull/645)) [@robertmaynard](https://github.com/robertmaynard)
- Set CUDA_RUNTIME_LIBRARY to documented case style ([#641](https://github.com/rapidsai/rapids-cmake/pull/641)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- rapids_test_install_relocatable now handles SOVERSION target properties ([#636](https://github.com/rapidsai/rapids-cmake/pull/636)) [@robertmaynard](https://github.com/robertmaynard)
- Eval CMAKE_CUDA_ARCHITECTURES before ENV{CUDAARCHS} ([#624](https://github.com/rapidsai/rapids-cmake/pull/624)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Fix CPM package docs. ([#637](https://github.com/rapidsai/rapids-cmake/pull/637)) [@bdice](https://github.com/bdice)
- expand rapids-cmake cpm docs ([#613](https://github.com/rapidsai/rapids-cmake/pull/613)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Add rapids_cpm_bs_thread_pool() ([#651](https://github.com/rapidsai/rapids-cmake/pull/651)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- proprietary_binary_cuda_version_mapping allows for better CUDA version mapping ([#648](https://github.com/rapidsai/rapids-cmake/pull/648)) [@robertmaynard](https://github.com/robertmaynard)
- use latest cuco with insert_or_apply CAS fix ([#646](https://github.com/rapidsai/rapids-cmake/pull/646)) [@srinivasyadav18](https://github.com/srinivasyadav18)
- Fetch the latest cuco with CAS fixes ([#644](https://github.com/rapidsai/rapids-cmake/pull/644)) [@PointKernel](https://github.com/PointKernel)
- Update to CPM v0.40.0 ([#642](https://github.com/rapidsai/rapids-cmake/pull/642)) [@robertmaynard](https://github.com/robertmaynard)
- Remove CCCL patch for PR 211. ([#640](https://github.com/rapidsai/rapids-cmake/pull/640)) [@bdice](https://github.com/bdice)
- Try any failing tests up to 3 times to guard against network issues ([#639](https://github.com/rapidsai/rapids-cmake/pull/639)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cmake_support_conda_env adds `-O0` to debug compile lines ([#635](https://github.com/rapidsai/rapids-cmake/pull/635)) [@robertmaynard](https://github.com/robertmaynard)
- Update cuco git tag to fetch new multiset data structure ([#628](https://github.com/rapidsai/rapids-cmake/pull/628)) [@PointKernel](https://github.com/PointKernel)
- Move required CMake version to 3.26.4 ([#627](https://github.com/rapidsai/rapids-cmake/pull/627)) [@robertmaynard](https://github.com/robertmaynard)
- Removes legacy rapids-cmake cython implementations as it is deprecated in 24.08 ([#614](https://github.com/rapidsai/rapids-cmake/pull/614)) [@robertmaynard](https://github.com/robertmaynard)
- Update CCCL to v2.5.0 ([#607](https://github.com/rapidsai/rapids-cmake/pull/607)) [@trxcllnt](https://github.com/trxcllnt)

## 🛠️ Improvements

- Use workflow branch 24.08 again ([#647](https://github.com/rapidsai/rapids-cmake/pull/647)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Build and test with CUDA 12.5.1 ([#643](https://github.com/rapidsai/rapids-cmake/pull/643)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Bump CCCL version to include cuda::std::span fix ([#631](https://github.com/rapidsai/rapids-cmake/pull/631)) [@sleeepyjack](https://github.com/sleeepyjack)
- Adopt CI/packaging codeowners ([#629](https://github.com/rapidsai/rapids-cmake/pull/629)) [@bdice](https://github.com/bdice)

# rapids-cmake 24.06.00 (5 Jun 2024)

## 🐛 Bug Fixes

- Only output CUDA architectures we are building for once ([#621](https://github.com/rapidsai/rapids-cmake/pull/621)) [@robertmaynard](https://github.com/robertmaynard)
- set_architectures output correct CUDA arch values for RAPIDS mode ([#619](https://github.com/rapidsai/rapids-cmake/pull/619)) [@robertmaynard](https://github.com/robertmaynard)
- Always offer the install target names for nvtx3 ([#617](https://github.com/rapidsai/rapids-cmake/pull/617)) [@robertmaynard](https://github.com/robertmaynard)
- Support CMAKE_INSTALL_MESSAGE + rapids_test_install_relocatable ([#604](https://github.com/rapidsai/rapids-cmake/pull/604)) [@robertmaynard](https://github.com/robertmaynard)
- Ensure nvcomps build and install layouts are consistent ([#602](https://github.com/rapidsai/rapids-cmake/pull/602)) [@robertmaynard](https://github.com/robertmaynard)
- Correctly set the install location for nvcomp when using the proprietary binary ([#597](https://github.com/rapidsai/rapids-cmake/pull/597)) [@vyasr](https://github.com/vyasr)
- Ensure support_conda_env uses `isystem` ([#588](https://github.com/rapidsai/rapids-cmake/pull/588)) [@robertmaynard](https://github.com/robertmaynard)
- Update rapids_test_install_relocatable to be aware of CMake 3.29 ([#586](https://github.com/rapidsai/rapids-cmake/pull/586)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cpm_gtest(BUILD_STATIC) now doesn&#39;t search for a local version ([#585](https://github.com/rapidsai/rapids-cmake/pull/585)) [@robertmaynard](https://github.com/robertmaynard)
- Add new patch to hide more CCCL APIs ([#580](https://github.com/rapidsai/rapids-cmake/pull/580)) [@vyasr](https://github.com/vyasr)
- Forward-merge branch-24.04 into branch-24.06 [skip ci] ([#565](https://github.com/rapidsai/rapids-cmake/pull/565)) [@rapids-bot[bot]](https://github.com/rapids-bot[bot])

## 📖 Documentation

- use inline code formatting in docs for variables, functions, and modules ([#591](https://github.com/rapidsai/rapids-cmake/pull/591)) [@jameslamb](https://github.com/jameslamb)
- clarify language around how FetchContent is used in RAPIDS.cmake ([#590](https://github.com/rapidsai/rapids-cmake/pull/590)) [@jameslamb](https://github.com/jameslamb)
- fix docs for rapids_export_ functions ([#589](https://github.com/rapidsai/rapids-cmake/pull/589)) [@jameslamb](https://github.com/jameslamb)
- Better explain OVERRIDE option ([#578](https://github.com/rapidsai/rapids-cmake/pull/578)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Output what cuda archs rapids-cmake is building for ([#609](https://github.com/rapidsai/rapids-cmake/pull/609)) [@robertmaynard](https://github.com/robertmaynard)
- Add rapids_cpm_nvtx3. ([#606](https://github.com/rapidsai/rapids-cmake/pull/606)) [@bdice](https://github.com/bdice)
- Refactor the common `verify` cpm pin test logic to a single source ([#601](https://github.com/rapidsai/rapids-cmake/pull/601)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake allow GENERATE_PINNED_VERSIONS via CMake variable ([#600](https://github.com/rapidsai/rapids-cmake/pull/600)) [@robertmaynard](https://github.com/robertmaynard)
- Allow for cpm default and override files via variables ([#596](https://github.com/rapidsai/rapids-cmake/pull/596)) [@robertmaynard](https://github.com/robertmaynard)
- Expand rapids_cpm_init to support custom default version files ([#595](https://github.com/rapidsai/rapids-cmake/pull/595)) [@robertmaynard](https://github.com/robertmaynard)
- Bump NVBench version for new `main` hooks. ([#584](https://github.com/rapidsai/rapids-cmake/pull/584)) [@alliepiper](https://github.com/alliepiper)
- `rapids_cython_create_modules()`: Generate Cython dependency file ([#579](https://github.com/rapidsai/rapids-cmake/pull/579)) [@Jacobfaib](https://github.com/Jacobfaib)
- rapids_cpm_gtest adds support for BUILD_STATIC ([#576](https://github.com/rapidsai/rapids-cmake/pull/576)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Fix `nvtx3` build export ([#615](https://github.com/rapidsai/rapids-cmake/pull/615)) [@trxcllnt](https://github.com/trxcllnt)
- limit pinning tests to CPM-downloaded projects ([#599](https://github.com/rapidsai/rapids-cmake/pull/599)) [@jameslamb](https://github.com/jameslamb)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#594](https://github.com/rapidsai/rapids-cmake/pull/594)) [@hcho3](https://github.com/hcho3)
- resolve &#39;file_key&#39; deprecation warning from rapids-dependency-file-generator ([#593](https://github.com/rapidsai/rapids-cmake/pull/593)) [@jameslamb](https://github.com/jameslamb)
- Prevent path conflict in builds ([#571](https://github.com/rapidsai/rapids-cmake/pull/571)) [@AyodeAwe](https://github.com/AyodeAwe)
- Bump cuco version to fetch the latest set retrieve API ([#569](https://github.com/rapidsai/rapids-cmake/pull/569)) [@PointKernel](https://github.com/PointKernel)
- Forward-merge branch-24.04 to branch-24.06 ([#563](https://github.com/rapidsai/rapids-cmake/pull/563)) [@bdice](https://github.com/bdice)

# rapids-cmake 24.04.00 (10 Apr 2024)

## 🐛 Bug Fixes

- nvcomp try proprietary binary when &#39;always_download&#39; is on ([#570](https://github.com/rapidsai/rapids-cmake/pull/570)) [@robertmaynard](https://github.com/robertmaynard)
- Update pre-commit-hooks to v0.0.3 ([#566](https://github.com/rapidsai/rapids-cmake/pull/566)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Always link against cudart_static ([#564](https://github.com/rapidsai/rapids-cmake/pull/564)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Use CUDA compiler if available for generate_resource_spec ([#561](https://github.com/rapidsai/rapids-cmake/pull/561)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- rapids-cmake support empty patches array in json ([#559](https://github.com/rapidsai/rapids-cmake/pull/559)) [@robertmaynard](https://github.com/robertmaynard)
- Handle CMake 3.28 new EXCLUDE_FROM_ALL option of `FetchContent` ([#557](https://github.com/rapidsai/rapids-cmake/pull/557)) [@robertmaynard](https://github.com/robertmaynard)
- Add rapids_cuda_set_runtime to default includes of cuda ([#538](https://github.com/rapidsai/rapids-cmake/pull/538)) [@robertmaynard](https://github.com/robertmaynard)
- add rapids-dependency-file-generator pre-commit hook ([#531](https://github.com/rapidsai/rapids-cmake/pull/531)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Deprecate rapids_cpm_libcudacxx and rapids_cpm_thrust. ([#560](https://github.com/rapidsai/rapids-cmake/pull/560)) [@bdice](https://github.com/bdice)
- rapids_cpm_package_details now validates required entries exist ([#558](https://github.com/rapidsai/rapids-cmake/pull/558)) [@robertmaynard](https://github.com/robertmaynard)
- Support getting rapids-cmake via git clone ([#555](https://github.com/rapidsai/rapids-cmake/pull/555)) [@robertmaynard](https://github.com/robertmaynard)
- Bump nvbench version for faster benchmark runs ([#549](https://github.com/rapidsai/rapids-cmake/pull/549)) [@PointKernel](https://github.com/PointKernel)
- Remove unneeded whitespace from json ([#544](https://github.com/rapidsai/rapids-cmake/pull/544)) [@robertmaynard](https://github.com/robertmaynard)
- Officially support env var expansion in version.json ([#540](https://github.com/rapidsai/rapids-cmake/pull/540)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake can generate pinned versions file ([#530](https://github.com/rapidsai/rapids-cmake/pull/530)) [@robertmaynard](https://github.com/robertmaynard)
- Fetch the latest cuco and remove outdated patches ([#526](https://github.com/rapidsai/rapids-cmake/pull/526)) [@PointKernel](https://github.com/PointKernel)
- Support CUDA 12.2 ([#521](https://github.com/rapidsai/rapids-cmake/pull/521)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#573](https://github.com/rapidsai/rapids-cmake/pull/573)) [@bdice](https://github.com/bdice)
- Replace local copyright check with pre-commit-hooks verify-copyright ([#556](https://github.com/rapidsai/rapids-cmake/pull/556)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add patch to fix fmt `v10.1.1` version ([#548](https://github.com/rapidsai/rapids-cmake/pull/548)) [@trxcllnt](https://github.com/trxcllnt)
- Add support for Python 3.11 ([#547](https://github.com/rapidsai/rapids-cmake/pull/547)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-24.02 to branch-24.04 ([#545](https://github.com/rapidsai/rapids-cmake/pull/545)) [@bdice](https://github.com/bdice)
- target branch-24.04 for GitHub Actions workflows ([#541](https://github.com/rapidsai/rapids-cmake/pull/541)) [@jameslamb](https://github.com/jameslamb)
- Build generate_ctest_json in try_compile() ([#537](https://github.com/rapidsai/rapids-cmake/pull/537)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Ensure that `ctest` is called with `--no-tests=error`. ([#535](https://github.com/rapidsai/rapids-cmake/pull/535)) [@bdice](https://github.com/bdice)
- Update ops-bot.yaml ([#532](https://github.com/rapidsai/rapids-cmake/pull/532)) [@AyodeAwe](https://github.com/AyodeAwe)

# rapids-cmake 24.02.00 (12 Feb 2024)

## 🚨 Breaking Changes

- Drop Pascal architecture (60). ([#482](https://github.com/rapidsai/rapids-cmake/pull/482)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- Error out if generate_ctest_json fails to build or run ([#533](https://github.com/rapidsai/rapids-cmake/pull/533)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- rapids_cpm_cccl now works as expected when given `DOWNLOAD_ONLY ON` ([#527](https://github.com/rapidsai/rapids-cmake/pull/527)) [@robertmaynard](https://github.com/robertmaynard)
- Always download repos when they are being patched ([#525](https://github.com/rapidsai/rapids-cmake/pull/525)) [@vyasr](https://github.com/vyasr)
- Mark all cccl and cuco kernels with hidden visibility ([#523](https://github.com/rapidsai/rapids-cmake/pull/523)) [@robertmaynard](https://github.com/robertmaynard)
- Fix message context ([#520](https://github.com/rapidsai/rapids-cmake/pull/520)) [@vyasr](https://github.com/vyasr)
- Generate template copyright year at build time. ([#325) (#519](https://github.com/rapidsai/rapids-cmake/pull/325) (#519)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add link libraries to generate_resource_spec cmake ([#516](https://github.com/rapidsai/rapids-cmake/pull/516)) [@davidwendt](https://github.com/davidwendt)
- rapids_cpm_cccl preserve install location details from first invocation. ([#513](https://github.com/rapidsai/rapids-cmake/pull/513)) [@robertmaynard](https://github.com/robertmaynard)
- Only apply install rules for CCCL if we actually downloaded ([#507](https://github.com/rapidsai/rapids-cmake/pull/507)) [@bdice](https://github.com/bdice)
- Mark flaky test as serial ([#506](https://github.com/rapidsai/rapids-cmake/pull/506)) [@vyasr](https://github.com/vyasr)
- Manually invoke install rules for components ([#505](https://github.com/rapidsai/rapids-cmake/pull/505)) [@vyasr](https://github.com/vyasr)
- multiple entry overrides now sets FetchContent for all entries ([#494](https://github.com/rapidsai/rapids-cmake/pull/494)) [@robertmaynard](https://github.com/robertmaynard)
- Remove deprecated function usages ([#484](https://github.com/rapidsai/rapids-cmake/pull/484)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Fix docs references to API sections. ([#509](https://github.com/rapidsai/rapids-cmake/pull/509)) [@bdice](https://github.com/bdice)
- fix typo in README ([#501](https://github.com/rapidsai/rapids-cmake/pull/501)) [@jameslamb](https://github.com/jameslamb)
- Fix indentation typo ([#497](https://github.com/rapidsai/rapids-cmake/pull/497)) [@vyasr](https://github.com/vyasr)

## 🚀 New Features

- rapids cpm patches now support differences in white space. ([#515](https://github.com/rapidsai/rapids-cmake/pull/515)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade nvCOMP to 3.0.5 ([#498](https://github.com/rapidsai/rapids-cmake/pull/498)) [@davidwendt](https://github.com/davidwendt)
- Move to latest nvbench which has nvml+static support ([#488](https://github.com/rapidsai/rapids-cmake/pull/488)) [@robertmaynard](https://github.com/robertmaynard)
- Update to spdlog 1.12 and fmt 10.1.1 ([#473](https://github.com/rapidsai/rapids-cmake/pull/473)) [@kkraus14](https://github.com/kkraus14)
- Support scikit-build-core ([#433](https://github.com/rapidsai/rapids-cmake/pull/433)) [@vyasr](https://github.com/vyasr)

## 🛠️ Improvements

- Remove usages of rapids-env-update ([#524](https://github.com/rapidsai/rapids-cmake/pull/524)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor CUDA versions in dependencies.yaml ([#517](https://github.com/rapidsai/rapids-cmake/pull/517)) [@jameslamb](https://github.com/jameslamb)
- Remove scikit-build from dependency list ([#512](https://github.com/rapidsai/rapids-cmake/pull/512)) [@vyasr](https://github.com/vyasr)
- Add patch reverting CCCL PR 211. ([#511](https://github.com/rapidsai/rapids-cmake/pull/511)) [@bdice](https://github.com/bdice)
- Update cuCollections for CCCL 2.2.0 support. ([#510](https://github.com/rapidsai/rapids-cmake/pull/510)) [@bdice](https://github.com/bdice)
- Disable NVBench CUPTI support by default. ([#504](https://github.com/rapidsai/rapids-cmake/pull/504)) [@bdice](https://github.com/bdice)
- Remove CCCL::Thrust from GLOBAL_TARGETS. ([#500](https://github.com/rapidsai/rapids-cmake/pull/500)) [@bdice](https://github.com/bdice)
- Add missing nvcomp targets ([#496](https://github.com/rapidsai/rapids-cmake/pull/496)) [@vyasr](https://github.com/vyasr)
- Add rapids_cpm_cccl feature. ([#495](https://github.com/rapidsai/rapids-cmake/pull/495)) [@bdice](https://github.com/bdice)
- Drop Pascal architecture (60). ([#482](https://github.com/rapidsai/rapids-cmake/pull/482)) [@bdice](https://github.com/bdice)

# rapids-cmake 23.12.00 (6 Dec 2023)

## 🚨 Breaking Changes

- Upgrade nvCOMP to 3.0.4 ([#451](https://github.com/rapidsai/rapids-cmake/pull/451)) [@vuule](https://github.com/vuule)

## 🐛 Bug Fixes

- Ensure nvbench initializes nvml context when built statically ([#486](https://github.com/rapidsai/rapids-cmake/pull/486)) [@robertmaynard](https://github.com/robertmaynard)
- Remove invalid argument to find_package_root ([#483](https://github.com/rapidsai/rapids-cmake/pull/483)) [@robertmaynard](https://github.com/robertmaynard)
- target from write_git_revision now works with export sets ([#474](https://github.com/rapidsai/rapids-cmake/pull/474)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Support static builds of gbench and nvbench. ([#481](https://github.com/rapidsai/rapids-cmake/pull/481)) [@robertmaynard](https://github.com/robertmaynard)
- Allow rapids_test to be used without CUDAToolkit ([#480](https://github.com/rapidsai/rapids-cmake/pull/480)) [@robertmaynard](https://github.com/robertmaynard)
- Update cuco git tag ([#479](https://github.com/rapidsai/rapids-cmake/pull/479)) [@sleeepyjack](https://github.com/sleeepyjack)
- GTest will always be PIC enabled when built ([#477](https://github.com/rapidsai/rapids-cmake/pull/477)) [@robertmaynard](https://github.com/robertmaynard)
- Thrust when exported now automatically calls `thrust_create_target` ([#467](https://github.com/rapidsai/rapids-cmake/pull/467)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade nvCOMP to 3.0.4 ([#451](https://github.com/rapidsai/rapids-cmake/pull/451)) [@vuule](https://github.com/vuule)

## 🛠️ Improvements

- Build concurrency for nightly and merge triggers ([#490](https://github.com/rapidsai/rapids-cmake/pull/490)) [@bdice](https://github.com/bdice)
- Add patch for libcudacxx memory resource. ([#476](https://github.com/rapidsai/rapids-cmake/pull/476)) [@bdice](https://github.com/bdice)
- Use branch-23.12 workflows. ([#472](https://github.com/rapidsai/rapids-cmake/pull/472)) [@bdice](https://github.com/bdice)
- Express Python version in dependencies.yaml. ([#470](https://github.com/rapidsai/rapids-cmake/pull/470)) [@bdice](https://github.com/bdice)
- Build CUDA 12.0 ARM conda packages. ([#468](https://github.com/rapidsai/rapids-cmake/pull/468)) [@bdice](https://github.com/bdice)
- Update libcudacxx to 2.1.0 ([#464](https://github.com/rapidsai/rapids-cmake/pull/464)) [@bdice](https://github.com/bdice)

# rapids-cmake 23.10.00 (11 Oct 2023)

## 🐛 Bug Fixes

- Quote the list of patch files in case they have spaces in their paths ([#463](https://github.com/rapidsai/rapids-cmake/pull/463)) [@ericniebler](https://github.com/ericniebler)
- cpm overrides don&#39;t occur when `CPM_&lt;pkg&gt;_SOURCE` exists ([#458](https://github.com/rapidsai/rapids-cmake/pull/458)) [@robertmaynard](https://github.com/robertmaynard)
- Use `conda mambabuild` not `mamba mambabuild` ([#457](https://github.com/rapidsai/rapids-cmake/pull/457)) [@bdice](https://github.com/bdice)
- Support fmt use in debug builds ([#456](https://github.com/rapidsai/rapids-cmake/pull/456)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Move rapids_cpm_package_override to CPM section of docs ([#462](https://github.com/rapidsai/rapids-cmake/pull/462)) [@robertmaynard](https://github.com/robertmaynard)
- Improve docs around fetch content and rapids-cmake overrides ([#444](https://github.com/rapidsai/rapids-cmake/pull/444)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Bump cuco version ([#452](https://github.com/rapidsai/rapids-cmake/pull/452)) [@PointKernel](https://github.com/PointKernel)

## 🛠️ Improvements

- Update image names ([#461](https://github.com/rapidsai/rapids-cmake/pull/461)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update to CPM v0.38.5 ([#460](https://github.com/rapidsai/rapids-cmake/pull/460)) [@trxcllnt](https://github.com/trxcllnt)
- Update to clang 16.0.6. ([#459](https://github.com/rapidsai/rapids-cmake/pull/459)) [@bdice](https://github.com/bdice)
- Use `copy-pr-bot` ([#455](https://github.com/rapidsai/rapids-cmake/pull/455)) [@ajschmidt8](https://github.com/ajschmidt8)

# rapids-cmake 23.08.00 (9 Aug 2023)

## 🐛 Bug Fixes

- Use &lt; gcc-11 with cuda 11.5 to avoid nvbench compile failures ([#448](https://github.com/rapidsai/rapids-cmake/pull/448)) [@robertmaynard](https://github.com/robertmaynard)
- Ensure tests the modify same git repo don&#39;t execute at the same time ([#446](https://github.com/rapidsai/rapids-cmake/pull/446)) [@robertmaynard](https://github.com/robertmaynard)
- Fix CUDA 11.5 tests by adding dependencies entries. ([#443](https://github.com/rapidsai/rapids-cmake/pull/443)) [@bdice](https://github.com/bdice)
- Remove trailing comma and add pre-commit hook for JSON validation. ([#440](https://github.com/rapidsai/rapids-cmake/pull/440)) [@bdice](https://github.com/bdice)
- When nvcomp is found locally print where it is on disk ([#434](https://github.com/rapidsai/rapids-cmake/pull/434)) [@robertmaynard](https://github.com/robertmaynard)
- Correct two issues found when testing CMake 3.27 rc2 ([#432](https://github.com/rapidsai/rapids-cmake/pull/432)) [@robertmaynard](https://github.com/robertmaynard)
- Correct re-root controls from conda-forge with thrust/cub/etc ([#431](https://github.com/rapidsai/rapids-cmake/pull/431)) [@robertmaynard](https://github.com/robertmaynard)
- Bug/proprietary binary obeys `always_download` ([#430](https://github.com/rapidsai/rapids-cmake/pull/430)) [@robertmaynard](https://github.com/robertmaynard)
- Correct install_relocatable issues found by libcudf ([#423](https://github.com/rapidsai/rapids-cmake/pull/423)) [@robertmaynard](https://github.com/robertmaynard)
- test_install_relocatable correct run_gpu_test.cmake location ([#420](https://github.com/rapidsai/rapids-cmake/pull/420)) [@robertmaynard](https://github.com/robertmaynard)
- Fea/move to latest nvbench ([#417](https://github.com/rapidsai/rapids-cmake/pull/417)) [@robertmaynard](https://github.com/robertmaynard)
- Use [@loader_path instead of $ORIGIN on MacOS ([#403](https://github.com/rapidsai/rapids-cmake/pull/403)) @manopapad](https://github.com/loader_path instead of $ORIGIN on MacOS ([#403](https://github.com/rapidsai/rapids-cmake/pull/403)) @manopapad)
- Make NAMESPACE property truly optional in rapids_export ([#358](https://github.com/rapidsai/rapids-cmake/pull/358)) [@agirault](https://github.com/agirault)

## 🚀 New Features

- Update rapids-cmake ci to support conda-forge CUDA 12 ([#437](https://github.com/rapidsai/rapids-cmake/pull/437)) [@robertmaynard](https://github.com/robertmaynard)
- Bump cuco version ([#435](https://github.com/rapidsai/rapids-cmake/pull/435)) [@PointKernel](https://github.com/PointKernel)
- Add rapids_cuda_set_runtime ([#429](https://github.com/rapidsai/rapids-cmake/pull/429)) [@robertmaynard](https://github.com/robertmaynard)
- support_conda_env support host and build CTK 12 locations ([#428](https://github.com/rapidsai/rapids-cmake/pull/428)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_find_generate_module Support user code blocks ([#415](https://github.com/rapidsai/rapids-cmake/pull/415)) [@robertmaynard](https://github.com/robertmaynard)
- Rewrite of rapids_test_install_relocatable to support genex expressions ([#410](https://github.com/rapidsai/rapids-cmake/pull/410)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Conditionally modify envvar vs. global CMAKE_PREFIX_PATH in `rapids_cmake_support_conda_env` ([#439](https://github.com/rapidsai/rapids-cmake/pull/439)) [@trxcllnt](https://github.com/trxcllnt)
- Migrate to updated shared-action-workflows name for CUDA 12 CI ([#438](https://github.com/rapidsai/rapids-cmake/pull/438)) [@bdice](https://github.com/bdice)
- Fix google benchmark name and update version ([#425](https://github.com/rapidsai/rapids-cmake/pull/425)) [@vyasr](https://github.com/vyasr)
- use rapids-upload-docs script ([#419](https://github.com/rapidsai/rapids-cmake/pull/419)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove documentation build scripts for Jenkins ([#418](https://github.com/rapidsai/rapids-cmake/pull/418)) [@ajschmidt8](https://github.com/ajschmidt8)
- Upload conda packages for rapids_core_dependencies. ([#414](https://github.com/rapidsai/rapids-cmake/pull/414)) [@bdice](https://github.com/bdice)

# rapids-cmake 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Using deprecated CUDA_ARCHITECTURE values now produces an error. ([#397](https://github.com/rapidsai/rapids-cmake/pull/397)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cpm cccl packages cmake files are now relocated to not clash with upstream ([#393](https://github.com/rapidsai/rapids-cmake/pull/393)) [@robertmaynard](https://github.com/robertmaynard)

## 🐛 Bug Fixes

- Revert &quot;Define Cython language_level explicitly. ([#394)&quot; (#396](https://github.com/rapidsai/rapids-cmake/pull/394)&quot; (#396)) [@vyasr](https://github.com/vyasr)
- rapids_cpm cccl packages cmake files are now relocated to not clash with upstream ([#393](https://github.com/rapidsai/rapids-cmake/pull/393)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Correct basics to api cross refs ([#405](https://github.com/rapidsai/rapids-cmake/pull/405)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Update cuco git tag to support `cuco::static_set` ([#407](https://github.com/rapidsai/rapids-cmake/pull/407)) [@PointKernel](https://github.com/PointKernel)
- Upgrade GTest version to 1.13 ([#401](https://github.com/rapidsai/rapids-cmake/pull/401)) [@robertmaynard](https://github.com/robertmaynard)
- Using deprecated CUDA_ARCHITECTURE values now produces an error. ([#397](https://github.com/rapidsai/rapids-cmake/pull/397)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- run docs nightly too ([#413](https://github.com/rapidsai/rapids-cmake/pull/413)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update cuco git tag to fetch several bug fixes ([#412](https://github.com/rapidsai/rapids-cmake/pull/412)) [@PointKernel](https://github.com/PointKernel)
- Revert shared workflows branch ([#406](https://github.com/rapidsai/rapids-cmake/pull/406)) [@ajschmidt8](https://github.com/ajschmidt8)
- Upgrade to Python 3.9 (drop Python 3.9) ([#404](https://github.com/rapidsai/rapids-cmake/pull/404)) [@shwina](https://github.com/shwina)
- Remove usage of rapids-get-rapids-version-from-git ([#402](https://github.com/rapidsai/rapids-cmake/pull/402)) [@jjacobelli](https://github.com/jjacobelli)
- Update clang-format ([#398](https://github.com/rapidsai/rapids-cmake/pull/398)) [@bdice](https://github.com/bdice)
- Define Cython language_level explicitly. ([#394](https://github.com/rapidsai/rapids-cmake/pull/394)) [@bdice](https://github.com/bdice)

# rapids-cmake 23.04.00 (6 Apr 2023)

## 🐛 Bug Fixes

- install_relocatable only installs files that exist ([#392](https://github.com/rapidsai/rapids-cmake/pull/392)) [@robertmaynard](https://github.com/robertmaynard)
- Revert &quot;install tests environment properties ([#390)&quot; (#391](https://github.com/rapidsai/rapids-cmake/pull/390)&quot; (#391)) [@robertmaynard](https://github.com/robertmaynard)
- Add `COMPONENT` arguments for rapids_export to formatting file. ([#389](https://github.com/rapidsai/rapids-cmake/pull/389)) [@robertmaynard](https://github.com/robertmaynard)
- install_relocatable generate correct installed RESOURCE_SPEC_FILE ([#386](https://github.com/rapidsai/rapids-cmake/pull/386)) [@robertmaynard](https://github.com/robertmaynard)
- support_conda_env only add rpath-link flag to linkers that support it. ([#384](https://github.com/rapidsai/rapids-cmake/pull/384)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cpm_nvbench properly specify usage of external fmt library ([#376](https://github.com/rapidsai/rapids-cmake/pull/376)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cpm_spdlog properly specify usage of external fmt library ([#375](https://github.com/rapidsai/rapids-cmake/pull/375)) [@robertmaynard](https://github.com/robertmaynard)
- Patch nvbench to allow usage of external fmt ([#373](https://github.com/rapidsai/rapids-cmake/pull/373)) [@robertmaynard](https://github.com/robertmaynard)
- Support static builds of fmt ([#372](https://github.com/rapidsai/rapids-cmake/pull/372)) [@robertmaynard](https://github.com/robertmaynard)
- Update to latest nvbench ([#371](https://github.com/rapidsai/rapids-cmake/pull/371)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Fix misspelling of rapids_cpm_init ([#385](https://github.com/rapidsai/rapids-cmake/pull/385)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- rapids_test_install_relocatable tracks tests environment properties ([#390](https://github.com/rapidsai/rapids-cmake/pull/390)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_test_install_relocatable EXCLUDE_FROM_ALL is now the default ([#388](https://github.com/rapidsai/rapids-cmake/pull/388)) [@robertmaynard](https://github.com/robertmaynard)
- Support downloading nvcomp CTK 11 or 12 binaries ([#381](https://github.com/rapidsai/rapids-cmake/pull/381)) [@robertmaynard](https://github.com/robertmaynard)
- Introduce clang-format to rapids-cmake to format C++ code examples ([#378](https://github.com/rapidsai/rapids-cmake/pull/378)) [@robertmaynard](https://github.com/robertmaynard)
- proprietary_binary now supports cuda toolkit version placeholders ([#377](https://github.com/rapidsai/rapids-cmake/pull/377)) [@robertmaynard](https://github.com/robertmaynard)
- Add `rapids_test` allowing projects to run gpu tests in parallel ([#328](https://github.com/rapidsai/rapids-cmake/pull/328)) [@robertmaynard](https://github.com/robertmaynard)
- Extend rapids_export to support the concept of optional COMPONENTS ([#154](https://github.com/rapidsai/rapids-cmake/pull/154)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Update to GCC 11 ([#382](https://github.com/rapidsai/rapids-cmake/pull/382)) [@bdice](https://github.com/bdice)
- Make docs builds less verbose ([#380](https://github.com/rapidsai/rapids-cmake/pull/380)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update GHAs Workflows ([#374](https://github.com/rapidsai/rapids-cmake/pull/374)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use trap to handle errors in test scripts ([#370](https://github.com/rapidsai/rapids-cmake/pull/370)) [@AjayThorve](https://github.com/AjayThorve)
- Bump spdlog to 1.11, add fmt as dependency for spdlog ([#368](https://github.com/rapidsai/rapids-cmake/pull/368)) [@kkraus14](https://github.com/kkraus14)
- Clean up and sort CPM packages. ([#366](https://github.com/rapidsai/rapids-cmake/pull/366)) [@bdice](https://github.com/bdice)
- Update shared workflow branches ([#365](https://github.com/rapidsai/rapids-cmake/pull/365)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add fmt 9.1.0 ([#364](https://github.com/rapidsai/rapids-cmake/pull/364)) [@kkraus14](https://github.com/kkraus14)
- Move date to build string in `conda` recipe ([#359](https://github.com/rapidsai/rapids-cmake/pull/359)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add docs build job ([#347](https://github.com/rapidsai/rapids-cmake/pull/347)) [@AyodeAwe](https://github.com/AyodeAwe)

# rapids-cmake 23.02.00 (9 Feb 2023)

## 🐛 Bug Fixes

- Remove incorrect deprecation for CMAKE_CUDA_ARCHITECTURES=&quot;NATIVE&quot; ([#355](https://github.com/rapidsai/rapids-cmake/pull/355)) [@robertmaynard](https://github.com/robertmaynard)
- cpm: `always_download` now considers `patches` json entry ([#353](https://github.com/rapidsai/rapids-cmake/pull/353)) [@robertmaynard](https://github.com/robertmaynard)
- Use string literals for policy test messages so no escaping needed ([#351](https://github.com/rapidsai/rapids-cmake/pull/351)) [@robertmaynard](https://github.com/robertmaynard)
- Revert &quot;Update spdlog to 1.11 ( latest version ) ([#342)&quot; (#346](https://github.com/rapidsai/rapids-cmake/pull/342)&quot; (#346)) [@bdice](https://github.com/bdice)
- Revert update of libcudacxx 1.9 ([#337](https://github.com/rapidsai/rapids-cmake/pull/337)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cuda_patch_toolkit: Better handle non-standard toolkits ([#324](https://github.com/rapidsai/rapids-cmake/pull/324)) [@robertmaynard](https://github.com/robertmaynard)
- Revert &quot;Upgrade spdlog to 1.10.0 ([#312)&quot; (#323](https://github.com/rapidsai/rapids-cmake/pull/312)&quot; (#323)) [@bdice](https://github.com/bdice)
- rapids_cuda_init_architectures now supports CUDAARCHS env variable ([#322](https://github.com/rapidsai/rapids-cmake/pull/322)) [@robertmaynard](https://github.com/robertmaynard)
- Remove usage of FetchContent from tests to improve perf ([#303](https://github.com/rapidsai/rapids-cmake/pull/303)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Update nvCOMP version to 2.6.1 ([#360](https://github.com/rapidsai/rapids-cmake/pull/360)) [@vuule](https://github.com/vuule)
- cpm: Rework `always_download` rules to be smarter ([#348](https://github.com/rapidsai/rapids-cmake/pull/348)) [@robertmaynard](https://github.com/robertmaynard)
- Add deprecation notice to passing &quot;&quot; to CMAKE_CUDA_ARCHITECTURES ([#345](https://github.com/rapidsai/rapids-cmake/pull/345)) [@robertmaynard](https://github.com/robertmaynard)
- Update to libcudacxx 1.9.1 to have a version &gt;= CUDA Toolkit 12 ([#343](https://github.com/rapidsai/rapids-cmake/pull/343)) [@robertmaynard](https://github.com/robertmaynard)
- Update spdlog to 1.11 ( latest version ) ([#342](https://github.com/rapidsai/rapids-cmake/pull/342)) [@robertmaynard](https://github.com/robertmaynard)
- Update to nvcomp 2.6 ([#341](https://github.com/rapidsai/rapids-cmake/pull/341)) [@robertmaynard](https://github.com/robertmaynard)
- Add deprecation warnings for usage of `ALL` ([#339](https://github.com/rapidsai/rapids-cmake/pull/339)) [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake now errors out when CPM can&#39;t be downloaded ([#335](https://github.com/rapidsai/rapids-cmake/pull/335)) [@robertmaynard](https://github.com/robertmaynard)
- Update to nvcomp 2.5 ([#333](https://github.com/rapidsai/rapids-cmake/pull/333)) [@robertmaynard](https://github.com/robertmaynard)
- Update to libcudacxx 1.9 to match version found in CUDA Toolkit 12 ([#332](https://github.com/rapidsai/rapids-cmake/pull/332)) [@robertmaynard](https://github.com/robertmaynard)
- Update cuco git tag to fetch bug fixes and cleanups ([#329](https://github.com/rapidsai/rapids-cmake/pull/329)) [@PointKernel](https://github.com/PointKernel)
- Fea/support cmake cuda architectures rapids value ([#327](https://github.com/rapidsai/rapids-cmake/pull/327)) [@robertmaynard](https://github.com/robertmaynard)
- Upgrade spdlog to 1.10.0 ([#312](https://github.com/rapidsai/rapids-cmake/pull/312)) [@kkraus14](https://github.com/kkraus14)

## 🛠️ Improvements

- Update shared workflow branches ([#361](https://github.com/rapidsai/rapids-cmake/pull/361)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build against CUDA `11.8` ([#344](https://github.com/rapidsai/rapids-cmake/pull/344)) [@ajschmidt8](https://github.com/ajschmidt8)
- Make generated find module targets global ([#340](https://github.com/rapidsai/rapids-cmake/pull/340)) [@vyasr](https://github.com/vyasr)
- Add codespell and whitespace linters to pre-commit hooks. ([#338](https://github.com/rapidsai/rapids-cmake/pull/338)) [@bdice](https://github.com/bdice)
- Use pre-commit for style checks ([#336](https://github.com/rapidsai/rapids-cmake/pull/336)) [@bdice](https://github.com/bdice)
- Branch 23.02 merge 22.12 ([#331](https://github.com/rapidsai/rapids-cmake/pull/331)) [@vyasr](https://github.com/vyasr)
- Update conda recipes. ([#330](https://github.com/rapidsai/rapids-cmake/pull/330)) [@bdice](https://github.com/bdice)
- Fix typo. ([#311](https://github.com/rapidsai/rapids-cmake/pull/311)) [@vyasr](https://github.com/vyasr)

# rapids-cmake 22.12.00 (8 Dec 2022)

## 🐛 Bug Fixes

- Don&#39;t use CMake 3.25.0 as it has a show stopping FindCUDAToolkit bug ([#308](https://github.com/rapidsai/rapids-cmake/pull/308)) [@robertmaynard](https://github.com/robertmaynard)
- Add missing CPM_ARGS to gbench ([#294](https://github.com/rapidsai/rapids-cmake/pull/294)) [@vyasr](https://github.com/vyasr)
- Patch results are only displayed once per invocation of CMake ([#292](https://github.com/rapidsai/rapids-cmake/pull/292)) [@robertmaynard](https://github.com/robertmaynard)
- Add thrust output iterator fix to rapids-cmake thrust patches ([#291](https://github.com/rapidsai/rapids-cmake/pull/291)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Update pull request template to match rest of RAPIDS ([#280](https://github.com/rapidsai/rapids-cmake/pull/280)) [@robertmaynard](https://github.com/robertmaynard)
- Clarify rapids_cuda_init_architectures behavior ([#279](https://github.com/rapidsai/rapids-cmake/pull/279)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Update cuco git tag ([#302](https://github.com/rapidsai/rapids-cmake/pull/302)) [@PointKernel](https://github.com/PointKernel)
- Remove old CI files ([#300](https://github.com/rapidsai/rapids-cmake/pull/300)) [@robertmaynard](https://github.com/robertmaynard)
- Update cuco to version that supports Ada and Hopper ([#299](https://github.com/rapidsai/rapids-cmake/pull/299)) [@robertmaynard](https://github.com/robertmaynard)
- Move libcudacxx 1.8.1 so we support sm90 ([#296](https://github.com/rapidsai/rapids-cmake/pull/296)) [@robertmaynard](https://github.com/robertmaynard)
- Add ability to specify library directories for target rpaths ([#295](https://github.com/rapidsai/rapids-cmake/pull/295)) [@vyasr](https://github.com/vyasr)
- Add support for cloning Google benchmark ([#293](https://github.com/rapidsai/rapids-cmake/pull/293)) [@vyasr](https://github.com/vyasr)
- Add `current_json_dir` placeholder in json patch file values ([#289](https://github.com/rapidsai/rapids-cmake/pull/289)) [@robertmaynard](https://github.com/robertmaynard)
- Add sm90 ( Hopper ) to rapids-cmake &quot;ALL&quot; mode ([#285](https://github.com/rapidsai/rapids-cmake/pull/285)) [@robertmaynard](https://github.com/robertmaynard)
- Enable copy_prs ops-bot config ([#284](https://github.com/rapidsai/rapids-cmake/pull/284)) [@robertmaynard](https://github.com/robertmaynard)
- Add GitHub action workflow to rapids-cmake ([#283](https://github.com/rapidsai/rapids-cmake/pull/283)) [@robertmaynard](https://github.com/robertmaynard)
- Create conda package of patched dependencies ([#275](https://github.com/rapidsai/rapids-cmake/pull/275)) [@robertmaynard](https://github.com/robertmaynard)
- Switch thrust over to use rapids-cmake patches ([#265](https://github.com/rapidsai/rapids-cmake/pull/265)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Remove `rapids-dependency-file-generator` `FIXME` ([#305](https://github.com/rapidsai/rapids-cmake/pull/305)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `ninja` as build dependency ([#301](https://github.com/rapidsai/rapids-cmake/pull/301)) [@ajschmidt8](https://github.com/ajschmidt8)
- Forward merge 22.10 into 22.12 ([#297](https://github.com/rapidsai/rapids-cmake/pull/297)) [@vyasr](https://github.com/vyasr)

# rapids-cmake 22.10.00 (12 Oct 2022)

## 🚨 Breaking Changes

- Update rapids-cmake to require cmake 3.23.1 (#227) @robertmaynard
- put $PREFIX before $BUILD_PREFIX in conda build (#182) @kkraus14

## 🐛 Bug Fixes

- Update to nvcomp 2.4.1 to fix zstd decompression (#286) @robertmaynard
- Restore rapids_cython_create_modules output variable name (#276) @robertmaynard
- rapids_cuda_init_architectures now obeys CUDAARCHS env variable (#270) @robertmaynard
- Update to Thrust 1.17.2 to fix cub ODR issues (#269) @robertmaynard
- conda_env: pass conda prefix as a rpath-link directory (#263) @robertmaynard
- Update cuCollections to fix issue with INSTALL_CUCO set to OFF. (#261) @bdice
- rapids_cpm_libcudacxx correct location of libcudacxx-config (#258) @robertmaynard
- Update rapids_find_generate_module to cmake 3.23 (#256) @robertmaynard
- Handle reconfiguring with USE_PROPRIETARY_BINARY value differing (#255) @robertmaynard
- rapids_cpm_thrust record build directory location of thrust-config (#254) @robertmaynard
- disable cuco install rules when no INSTALL_EXPORT_SET (#250) @robertmaynard
- Patch thrust and cub install rules to have proper header searches (#244) @robertmaynard
- Ensure that we install Thrust and Cub correctly. (#243) @robertmaynard
- Revert &quot;Update to CPM v0.35.4 for URL downloads... (#236)&quot; (#242) @robertmaynard
- put $PREFIX before $BUILD_PREFIX in conda build (#182) @kkraus14

## 📖 Documentation

- Correct broken patch_toolkit API docs, and CMake API cross references (#271) @robertmaynard
- Provide suggestions when encountering an incomplete GTest package (#247) @robertmaynard
- Docs: RAPIDS.cmake should be placed in current bin dir (#241) @robertmaynard
- Remove incorrect install location note on rapids_export (#232) @robertmaynard

## 🚀 New Features

- Update to CPM 0.35.6 as it has needed changes for cpm patching support. (#273) @robertmaynard
- Update to nvcomp 2.4 which now offers aarch64 binaries! (#272) @robertmaynard
- Support the concept of a patches to apply to a project built via CPM (#264) @robertmaynard
- Branch 22.10 merge 22.08 (#262) @robertmaynard
- Introduce rapids_cuda_patch_toolkit (#260) @robertmaynard
- Update libcudacxx to 1.8 (#253) @robertmaynard
- Update to CPM version 0.35.5 (#249) @robertmaynard
- Update to CPM v0.35.4 for URL downloads match the download time (#236) @robertmaynard
- rapids-cmake dependency tracking now understands COMPONENTS (#234) @robertmaynard
- Update to thrust 1.17 (#231) @robertmaynard
- Update to CPM v0.35.3 to support symlink build directories (#230) @robertmaynard
- Update rapids-cmake to require cmake 3.23.1 (#227) @robertmaynard
- Improve GPU detection by doing less subsequent executions (#222) @robertmaynard

## 🛠️ Improvements

- Fix typo in `rapids-cmake-url` (#267) @trxcllnt
- Ensure `&lt;pkg&gt;_FOUND` is set in the generated `Find&lt;pkg&gt;.cmake` file (#266) @trxcllnt
- Set `CUDA_USE_STATIC_CUDA_RUNTIME` to control legacy `FindCUDA.cmake`behavior (#259) @trxcllnt
- Use the GitHub `.zip` URI instead of `GIT_REPOSITORY` and `GIT_BRANCH` (#257) @trxcllnt
- Update nvcomp to 2.3.3 (#221) @vyasr

# rapids-cmake 22.08.00 (17 Aug 2022)

## 🐛 Bug Fixes

- json exclude flag behaves as expected libcudacx//thrust/nvcomp ([#223](https://github.com/rapidsai/rapids-cmake/pull/223)) [@robertmaynard](https://github.com/robertmaynard)
- Remove nvcomp dependency on CUDA::cudart_static ([#218](https://github.com/rapidsai/rapids-cmake/pull/218)) [@robertmaynard](https://github.com/robertmaynard)
- Timestamps for URL downloads match the download time ([#215](https://github.com/rapidsai/rapids-cmake/pull/215)) [@robertmaynard](https://github.com/robertmaynard)
- Revert &quot;Update nvcomp to 2.3.2 ([#209)&quot; (#210](https://github.com/rapidsai/rapids-cmake/pull/209)&quot; (#210)) [@vyasr](https://github.com/vyasr)
- rapids-cmake won&#39;t ever use an existing variable starting with RAPIDS_ ([#203](https://github.com/rapidsai/rapids-cmake/pull/203)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Docs now provide rapids_find_package examples ([#220](https://github.com/rapidsai/rapids-cmake/pull/220)) [@robertmaynard](https://github.com/robertmaynard)
- Minor typo fix in api.rst ([#207](https://github.com/rapidsai/rapids-cmake/pull/207)) [@vyasr](https://github.com/vyasr)
- rapids_cpm_&lt;pkgs&gt; document handling of unparsed args ([#206](https://github.com/rapidsai/rapids-cmake/pull/206)) [@robertmaynard](https://github.com/robertmaynard)
- Docs/remove doc warnings ([#205](https://github.com/rapidsai/rapids-cmake/pull/205)) [@robertmaynard](https://github.com/robertmaynard)
- Fix docs: default behavior is to use a shallow git clone. ([#204](https://github.com/rapidsai/rapids-cmake/pull/204)) [@bdice](https://github.com/bdice)
- Add rapids_cython to the html docs ([#197](https://github.com/rapidsai/rapids-cmake/pull/197)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- More robust solution of CMake policy 135 ([#224](https://github.com/rapidsai/rapids-cmake/pull/224)) [@robertmaynard](https://github.com/robertmaynard)
- Update cuco git tag ([#213](https://github.com/rapidsai/rapids-cmake/pull/213)) [@PointKernel](https://github.com/PointKernel)
- Revert &quot;Revert &quot;Update nvcomp to 2.3.2 ([#209)&quot; (#210)&quot; (#211](https://github.com/rapidsai/rapids-cmake/pull/209)&quot; (#210)&quot; (#211)) [@vyasr](https://github.com/vyasr)
- Update nvcomp to 2.3.2 ([#209](https://github.com/rapidsai/rapids-cmake/pull/209)) [@robertmaynard](https://github.com/robertmaynard)
- rapids_cpm_rmm no longer install when no INSTALL_EXPORT_SET listed ([#202](https://github.com/rapidsai/rapids-cmake/pull/202)) [@robertmaynard](https://github.com/robertmaynard)
- Adds support for pulling cuCollections using rapids-cmake ([#201](https://github.com/rapidsai/rapids-cmake/pull/201)) [@vyasr](https://github.com/vyasr)
- Add support for a prefix in Cython module targets ([#198](https://github.com/rapidsai/rapids-cmake/pull/198)) [@vyasr](https://github.com/vyasr)

## 🛠️ Improvements

- `rapids_find_package()` called with explicit version and REQUIRED should fail ([#214](https://github.com/rapidsai/rapids-cmake/pull/214)) [@trxcllnt](https://github.com/trxcllnt)

# rapids-cmake 22.06.00 (7 June 2022)

## 🐛 Bug Fixes

- nvcomp install rules need to match the pre-built layout (#194) @robertmaynard
- Use target name variable. (#187) @bdice
- Remove unneeded message from rapids_export_package (#183) @robertmaynard
- rapids_cpm_thrust: Correctly find version 1.15.0 (#181) @robertmaynard
- rapids_cpm_thrust: Correctly find version 1.15.0 (#180) @robertmaynard

## 📖 Documentation

- Correct spelling mistake in cpm package docs (#188) @robertmaynard

## 🚀 New Features

- Add rapids_cpm_nvcomp with prebuilt binary support (#190) @robertmaynard
- Default Cython module RUNPATH to $ORIGIN and return the list of created targets (#189) @vyasr
- Add rapids-cython component for scikit-build based Python package builds (#184) @vyasr
- Add more exhaustive set of tests are version values of 0 (#178) @robertmaynard
- rapids_cpm_package_override now hooks into FetchContent (#164) @robertmaynard

## 🛠️ Improvements

- Update nvbench tag (#193) @PointKernel

# rapids-cmake 22.04.00 (6 Apr 2022)

## 🐛 Bug Fixes

- rapids_export now handles explicit version values of 0 correctly (#174) @robertmaynard
- rapids_export now internally uses better named variables (#172) @robertmaynard
- rapids_cpm_gtest will properly find GTest 1.10 packages (#168) @robertmaynard
- CMAKE_CUDA_ARCHITECTURES `ALL` will not insert 62 or 72 (#161) @robertmaynard
- Tracked package versions are now not required, but preferred. (#160) @robertmaynard
- cpm_thrust would fail when provided only an install export set (#155) @robertmaynard
- rapids_export generated config.cmake no longer leaks variables (#149) @robertmaynard

## 📖 Documentation

- Docs use intersphinx correctly to link to CMake command docs (#159) @robertmaynard
- Example explains when you should use `rapids_find_generate_module` (#153) @robertmaynard
- Add CMake intersphinx support (#147) @bdice

## 🚀 New Features

- Bump CPM 0.35 for per package CPM_DOWNLOAD controls (#158) @robertmaynard
- Track package versions to the generated `find_dependency` calls (#156) @robertmaynard
- Update to latest nvbench (#150) @robertmaynard

## 🛠️ Improvements

- Temporarily disable new `ops-bot` functionality (#170) @ajschmidt8
- Use exact gtest version (#165) @trxcllnt
- Add `.github/ops-bot.yaml` config file (#163) @ajschmidt8

# rapids-cmake 22.02.00 (2 Feb 2022)

## 🐛 Bug Fixes

- Ensure that nvbench doesn&#39;t require nvml when `CUDA::nvml` doesn&#39;t exist (#146) @robertmaynard
- rapids_cpm_libcudacxx handle CPM already finding libcudacxx before being called (#130) @robertmaynard

## 📖 Documentation

- Fix typos (#142) @ajschmidt8
- Fix type-o in docs `&lt;PackageName&gt;_BINARY_DIR` instead of `&lt;PackageName&gt;_BINAR_DIR` (#140) @dagardner-nv
- Set the `always_download` value in versions.json to the common case (#135) @robertmaynard
- Update Changelog to capture all 21.08 and 21.10 changes (#134) @robertmaynard
- Correct minor formatting issues (#132) @robertmaynard
- Document how to control the git rep/tag that RAPIDS.cmake uses (#131) @robertmaynard

## 🚀 New Features

- rapids-cmake now supports an empty package entry in the override file (#145) @robertmaynard
- Update NVBench for 22.02 to be the latest version (#144) @robertmaynard
- Update rapids-cmake packages to libcudacxx 1.7 (#143) @robertmaynard
- Update rapids-cmake packages to Thrust 1.15 (#138) @robertmaynard
- add exclude_from_all flag to version.json (#137) @robertmaynard
- Add `PREFIX` option to write_version_file / write_git_revision_file (#118) @robertmaynard

## 🛠️ Improvements

- Remove rapids_cmake_install_lib_dir unstable side effect checks (#136) @robertmaynard

# rapids-cmake 21.12.00 (9 Dec 2021)

## 🐛 Bug Fixes

- rapids_cpm_libcudacxx install logic is safe for multiple inclusion (#124) @robertmaynard
- rapids_cpm_libcudacxx ensures CMAKE_INSTALL_INCLUDEDIR exists (#122) @robertmaynard
- rapids_cpm_find restores CPM variables when project was already added (#121) @robertmaynard
- rapids_cpm_thrust doesn&#39;t place temp file in a searched location (#120) @robertmaynard
- Require the exact version of Thrust in the versions.json file (#119) @trxcllnt
- CMake option second parameter is the help string, not the default value (#114) @robertmaynard
- Make sure we don&#39;t do a shallow clone on nvbench (#113) @robertmaynard
- Pin NVBench to a known working SHA1 (#112) @robertmaynard
- Build directory config.cmake now sets the correct targets to global (#110) @robertmaynard
- rapids_cpm_thrust installs to a location that won&#39;t be marked system (#98) @robertmaynard
- find_package now will find modules that CPM has downloaded. (#96) @robertmaynard
- rapids_cpm_thrust dont export namespaced thrust target (#93) @robertmaynard
- rapids_cpm_spdlog specifies the correct install variable (#91) @robertmaynard
- rapids_cpm_init: `CPM_SOURCE_CACHE` doesn&#39;t mean the CPM file exists (#87) @robertmaynard

## 📖 Documentation

- Better document that rapids_cpm_find supports arbitrary projects (#108) @robertmaynard
- Update the example to showcase rapids-cmake 21.12 (#107) @robertmaynard
- Properly generate rapids_cuda_init_runtime docs (#106) @robertmaynard

## 🚀 New Features

- Introduce rapids_cpm_libcudacxx (#111) @robertmaynard
- Record formatting rules for rapids_cpm_find DOWNLOAD_ONLY option (#94) @robertmaynard
- rapids_cmake_install_lib_dir now aware of GNUInstallDirs improvements in CMake 3.22 (#85) @robertmaynard
- rapids-cmake defaults to always download overridden packages (#83) @robertmaynard

## 🛠️ Improvements

- Prefer `CPM_&lt;pkg&gt;_SOURCE` dirs over `find_package()` in `rapids_cpm_find` (#92) @trxcllnt

# rapids-cmake 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- Remove unneeded inclusions of the old setup_cpm_cache.cmake (#82) @robertmaynard
- Make sure rapids-cmake doesn&#39;t produce CMake syntax warnings (#80) @robertmaynard
- rapids_export verify DOCUMENTATION and FINAL_CODE_BLOCK exist (#75) @robertmaynard
- Make sure rapids_cpm_spdlog specifies the correct spdlog global targets (#71) @robertmaynard
- rapids_cpm_thrust specifies the correct install variable (#70) @robertmaynard
- FIX Install sphinxcontrib-moderncmakedomain in docs script (#69) @dillon-cullinan
- rapids_export_cpm(BUILD) captures location of locally found packages (#65) @robertmaynard
- Introduce rapids_cmake_install_lib_dir (#61) @robertmaynard
- rapids_export(BUILD) only creates alias targets to existing targets (#55) @robertmaynard
- rapids_find_package propagates variables from find_package (#54) @robertmaynard
- rapids_cpm_find is more invariant as one would expect (#51) @robertmaynard
- rapids-cmake tests properly state what C++ std levels they require (#46) @robertmaynard
- rapids-cmake always generates GLOBAL_TARGETS names correctly (#36) @robertmaynard

## 📖 Documentation

- Update update-version.sh (#84) @raydouglass
- Add rapids_export_find_package_root to api doc page (#76) @robertmaynard
- README.md now references online docs (#72) @robertmaynard
- Copyright year range now matches when rapids-cmake existed (#67) @robertmaynard
- cmake-format: Now aware of `rapids_cmake_support_conda_env` flags (#62) @robertmaynard
- Bug/correct invalid generate module doc layout (#47) @robertmaynard

## 🚀 New Features

- rapids-cmake SHOULD_FAIL tests verify the CMake Error string (#79) @robertmaynard
- Introduce rapids_cmake_write_git_revision_file (#77) @robertmaynard
- Allow projects to override version.json information (#74) @robertmaynard
- rapids_export_package(BUILD) captures location of locally found packages (#68) @robertmaynard
- Introduce rapids_export_find_package_root command (#64) @robertmaynard
- Introduce rapids_cpm_&lt;preset&gt; (#52) @robertmaynard
- Tests now can be SERIAL and use FetchContent to get rapids-cmake (#48) @robertmaynard
- rapids_export version support expanded to handle more use-cases (#37) @robertmaynard

## 🛠️ Improvements

- cpm tests now download less components and can be run in parallel. (#81) @robertmaynard
- Ensure that all rapids-cmake files have include guards (#63) @robertmaynard
- Introduce RAPIDS.cmake a better way to fetch rapids-cmake (#45) @robertmaynard
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry (#44) @dillon-cullinan

# rapids-cmake 21.08.00 (4 Aug 2021)


## 🚀 New Features

- Introduce `rapids_cmake_write_version_file` to generate a C++ version header ([#23](https://github.com/rapidsai/rapids-cmake/pull/23)) [@robertmaynard](https://github.com/robertmaynard)
- Introduce `cmake-format-rapids-cmake` to allow `cmake-format` to understand rapdids-cmake custom functions ([#29](https://github.com/rapidsai/rapids-cmake/pull/29)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements


## 🐛 Bug Fixes

- ci/gpu/build.sh uses git tags to properly compute conda env (#43) @robertmaynard
- Make sure that rapids-cmake-dir cache variable is hidden (#40) @robertmaynard
- Correct regression specify rapids-cmake-dir as a cache variable (#39) @robertmaynard
- rapids-cmake add entries to CMAKE_MODULE_PATH on first config (#34) @robertmaynard
- Add tests that verify all paths in each rapids-<component>.cmake file ([#24](https://github.com/rapidsai/rapids-cmake/pull/24))  [@robertmaynard](https://github.com/robertmaynard)
- Correct issue where `rapids_export(DOCUMENTATION` content was being ignored([#30](https://github.com/rapidsai/rapids-cmake/pull/30))  [@robertmaynard](https://github.com/robertmaynard)
- rapids-cmake can now be correctly used by multiple adjacent directories ([#33](https://github.com/rapidsai/rapids-cmake/pull/33))  [@robertmaynard](https://github.com/robertmaynard)


# rapids-cmake 21.06.00 (Date TBD)

Please see https://github.com/rapidsai/rapids-cmake/releases/tag/v21.06.0a for the latest changes to this development branch.

## 🚀 New Features

- Introduce `rapids_cmake_parse_version` for better version extraction ([#20](https://github.com/rapidsai/rapids-cmake/pull/20)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Verify that rapids-cmake always preserves CPM arguments ([#18](https://github.com/rapidsai/rapids-cmake/pull/18))  [@robertmaynard](https://github.com/robertmaynard)
- Add Sphinx based documentation for the project  ([#14](https://github.com/rapidsai/rapids-cmake/pull/14))  [@robertmaynard](https://github.com/robertmaynard)
- `rapids_export` places the build export files in a location CPM can find. ([#3](https://github.com/rapidsai/rapids-cmake/pull/3))  [@robertmaynard](https://github.com/robertmaynard)

## 🐛 Bug Fixes

- Make sure we properly quote all CPM args ([#17](https://github.com/rapidsai/rapids-cmake/pull/17))  [@robertmaynard](https://github.com/robertmaynard)
- `rapids_export` correctly handles version strings with leading zeroes  ([#12](https://github.com/rapidsai/rapids-cmake/pull/12))  [@robertmaynard](https://github.com/robertmaynard)
- `rapids_export_write_language` properly executes each time CMake is run ([#10](https://github.com/rapidsai/rapids-cmake/pull/10))  [@robertmaynard](https://github.com/robertmaynard)
- `rapids_export` properly sets version variables ([#9](https://github.com/rapidsai/rapids-cmake/pull/9))  [@robertmaynard](https://github.com/robertmaynard)
- `rapids_export` now obeys CMake config file naming convention ([#8](https://github.com/rapidsai/rapids-cmake/pull/8))  [@robertmaynard](https://github.com/robertmaynard)
- Refactor layout to enable adding CI and Documentation ([#5](https://github.com/rapidsai/rapids-cmake/pull/5))  [@robertmaynard](https://github.com/robertmaynard)
