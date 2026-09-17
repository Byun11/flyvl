# FlyVL protocol (pre-registered)

이 파일은 CIFAR 데이터를 보기 전에 작성하고 commit한다. 이후 판정 기준을 바꾸면 새 커밋으로 남기고,
결과를 본 뒤의 변경이라는 사실을 명시한다.

## Question

> Does the frozen MaleCNS transform visual input into a representation in which object class is
> linearly decodable, and does the measured wiring outperform matched rewired controls?

결정적인 CNS는 label 정보를 새로 만들지 않는다. 측정 대상은 정보 증가가 아니라 linear accessibility이다.

## Model: FlyVL-simple-v1 (frozen, `configs/sim_frozen.json`)

| 항목 | 값 |
|---|---|
| topology / weights | MaleCNS v1.0, flybrain prebuilt `weights.npz` (synapse count × transmitter sign, 뉴런별 입력합 정규화), sha256 `c29919aa…40c` |
| optic lobe (`ol_intrinsic`, `ol_sensory`) | graded rate, rest 0.5, τ 40 ms, saturation 1.0 |
| central brain / VNC | LIF (flybrain: τ 100 ms, tonic 0.14, threshold 1) |
| gains | g_gg 4, g_gs 10, g_sg 0.5, g_ss 3 |
| noise | OFF |
| nonvisual sensory input | OFF (`cut_input_to_nonvisual_sensory`) |
| KC→KC | 0 (`kc_kc_scale`) |
| input | achromatic luminance → R1–6 (column 있는 3,241개), transduction `x = lum − A`, τ_adapt 0.5 s |
| presentation | gray warm-up 100 steps → 동일 snapshot에서 4-way symmetric drift (LR, RL, UD, DU), 50 steps each |
| feature | 뉴런별, 4 방향 평균의 (image branch − blank branch): LIF spike count, graded mean x |

이 이후 dynamics는 수정하지 않는다. v1 결과가 나온 뒤 motion dynamics 추가는 별도 모델(v2)로 비교한다.

## P0 결과 (CIFAR 이전)

- **P0-0** PASS: 166,700 neurons, 25,582,938 edges, 입력 행 |w| 합 = 1, sha 일치.
- **bench**: B=64에서 3.7 ms/step, peak 392 MB.
- **P0-1** (합성 자극, label 없음, grid 72 + 12 + 12 configs, 선택 규칙은 실행 전에 고정):

| 체크 | 결과 |
|---|---|
| Visual propagation (R1-6 → L1/L2 → Mi/Tm → T4/T5) | PASS |
| Looming pathway (LPLC2/LC4 → DNp01, ipsilateral) | PASS |
| Central CNS activation (P0-A eye route) | PASS (central 유닛 28–45%) |
| Detector diagnostic (P0-B) | PASS (LC4+LPLC2 L → DNp01 L 38 Hz) |
| T4/T5 direction tuning | **FAIL** (4–6/16, 우연 수준; known failure) |

설계 중 발견하여 명시한 수정: frozen noise는 KC 루프 폭주 → noise OFF; KC→KC axo-axonal 흥분 루프가
시각 자극으로 점화(LIF 5%가 50 Hz 고정) → `kc_kc_scale = 0`.

**T4/T5 direction selectivity is not required for progression to P1 because P1 evaluates static-image
class decodability, not motion computation.** T4/T5 DS는 gate가 아니라 diagnostic이다.
front_sign은 데이터로 결정되지 않아 +1로 고정하고, 4-way drift ensemble로 방향 편향을 제거한다.

## Controls

모든 control graph는 **effective graph** (`effective_W`: sensory cut + KC→KC 제거 후)에서 생성한다.
rewiring 조건: 동일 neuron set, 동일 edge 수(중복 없음), 모든 뉴런의 in/out-degree 보존, edge는
presynaptic 뉴런·weight·sign 유지, 뉴런별 입력 |w| 합을 effective graph와 동일하게 재조정,
KC→KC edge 생성 금지, self-loop 없음 (effective graph에는 101개, 무시할 수준).
- `global_shuffle`: 전체 edge에서 post 치환.
- `matched_shuffle`: (superclass×side of pre, superclass×side of post) 블록 안에서만 치환.
- readout mask는 real graph 기준 neuron ID로 한 번 정하고 모든 graph에 동일 적용.

## P0-2 gate (CIFAR-10 train, 클래스당 첫 10장 = 100장)

모두 만족하면 PASS → P1-mini, 하나라도 실패하면 STOP.
- G1 reproducibility: 같은 100장을 두 번 추출했을 때 LIF spike-count feature가 원소 99.99% 이상 동일하고,
  graded feature의 최대 절대 차이 ≤ 1e-4.
  (원래 bit-exact였으나 CIFAR 이전에 변경: cuSPARSE SpMM이 GPU에서 비결정적이라 반복 간 ≤2e-7 차이가 남.
  `use_deterministic_algorithms`와 float64로도 해결되지 않음. 노이즈 이미지 8장 스모크 테스트에서
  spike count는 100% 동일, graded 상대 L2 차이 5e-9. 결정적 SpMM 직접 구현은 약 10배 느려서 채택하지 않음.)
- G2 non-zero: 95% 이상의 이미지에서 `central_vnc` evoked feature가 0이 아님.
- G3 no explosion: 모든 이미지와 방향에서 마지막 10 step의 LIF 중 >40 Hz 비율 < 1%.
- G4 non-collapse (`central_vnc`): participation ratio ≥ 3 **그리고** 이미지 쌍 cosine distance 중앙값 ≥ 0.05.
참고용(판정에 쓰지 않음): 같은 지표의 pixels / stim_only 값, PCA 2D 그림.

## P1-mini (train 클래스당 첫 500장 = 5k, test 클래스당 첫 100장 = 1k)

- representations: `pixels` (32×32 luma), `stim_only` (R1-6 evoked = real feature의 driven 부분),
  `randproj` (pixels → 16,384-d Gaussian random projection + ReLU), `global_shuffle` seed 0, `real`.
- views: all, no_photoreceptor, central_vnc, visual_projection, descending.
- probe: standardize → train-only PCA to K ∈ {256, 1024, 4096} → multinomial logistic regression;
  L2 강도는 train 내부 5-fold CV로 선택. probe seed 3 (CV split). test accuracy + paired bootstrap 95% CI (1,000회).
- **primary comparison**: `real` vs `global_shuffle`, view `central_vnc`, K = 1024.
- 판정: primary 차이 (real − global_shuffle)의 point estimate ≤ 0이면 STOP. > 0이면 P1-full로 진행
  (CI는 보고만 하고 mini에서는 판정에 쓰지 않음).

## P1-full 판정 기준

필수: (1) eye → central non-zero evoked, (2) no collapse, (3) real central_vnc > chance,
(4) real > shuffled 방향이 반복됨.
강한 성공: real − matched_shuffle ≥ 3%p, 3 graph seeds 모두 같은 방향, test bootstrap CI 하한 > 0.
판단에는 PCA / effective rank / pairwise distance / probe accuracy만 사용; UMAP은 그림용.
