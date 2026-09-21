$$\begin{matrix} \text{GaitScheduler} & \longrightarrow & \text{FootstepPlanner / TrajectoryGenerator} & \longrightarrow & \mathbf{p}_{\text{foot, des}}, \mathbf{v}_{\text{foot, des}} \\ \downarrow & & \downarrow & & \downarrow \\ \text{MPCController} & \longrightarrow & f_{\text{mpc}} \longrightarrow \mathbf{WBICController} & \longrightarrow & \boldsymbol{\tau} \longrightarrow \mathbf{Plant} \end{matrix}$$

1. GaitScheduler wyznacza czas i fazy chodu.
2. FootstepPlanner + LegTrajectoryGenerator budują trajektorię stóp w fazie wymachu.
3. MPC wylicza pożądane siły dynamiki środka ciężkości / podparcia. 
4. WBIC spina wszystko w całość (kinematykę stóp + dynamikę MPC + priorytety zadań w null-space) i oblicza końcowy wektor momentów na przegubach $\tau$.


WBIC oblicza i wypluwa 3 zestawy sygnałów dla każdego stawu:Docelową pozycję stawu: $q_j^{\text{cmd}}$Docelową prędkość stawu: $\dot{q}_j^{\text{cmd}}$Wyceniony moment obrotowy (feedforward): $\tau_j$

ak fizycznie działa ten regulator niskopoziomowy?Sterownik wewnątrz każdego silnika (low-level joint actuator) przyjmuje komendy z WBIC i liczy ostateczny moment przekazywany na wał silnika według wzoru:$$\tau_{\text{motor}} = \tau_j + k_p \cdot \left(q_j^{\text{cmd}} - q_j\right) + k_d \cdot \left(\dot{q}_j^{\text{cmd}} - \dot{q}_j\right)$$$\tau_j$ – moment wyliczony przez WBIC (tzw. torque feedforward). Odpowiada za utrzymanie ciężaru ciała, dynamikę wymuszoną przez MPC oraz reakcję od podłoża.$k_p(q^{\text{cmd}} - q) + k_d(\dot{q}^{\text{cmd}} - \dot{q})$ – korekta PD w stawach. Jest szczególnie cenna dla nóg w powietrzu (swing legs), zapewniając precyzyjne podążanie stopy po zadanej trajektorii i zapobiegając uderzeniom czy drganiom.

---
- Wizja z lat 80. (Heurystyka i faza): GaitScheduler + FootstepPlanner (Raibert) dają intuicję, gdzie robot powinien postawić nogę na podstawie prędkości tułowia.
- Nowoczesność (MPC + WBIC):
    - Zamiast liczyć reakcję podłoża heurystycznie, MPC optymalizuje siły kontaktowe $f_{\text{mpc}}$ uwzględniając stożek tarcia.
    - Zamiast sterować przegubami na sztywno, WBIC rozwiąże dynamikę całego ciała w kaskadzie priorytetów (przestrzeń zerowa) – dzięki czemu robot np. przenosi wagę (MPC), jednocześnie wykonując płynny wymach nogi (kinematyka) i nie przekraczając dopuszczalnego momentu $\tau_{\text{max}}$.