# Minimal ABM for Perpetual Decentralized Storage

## Giới thiệu

Dự án này xây dựng một mô hình dựa trên tác tử (Agent-Based Model, ABM) để khảo sát khả năng duy trì dữ liệu trong một mạng lưu trữ phi tập trung. Mô hình tập trung vào tương tác giữa kinh tế của miner, giá lưu trữ, quyết định rời mạng, replica của dataset, cơ chế repair và Endowment của hệ thống.

Đây là **mô hình nghiên cứu ở giai đoạn tối giản**, chưa phải mô phỏng toàn bộ giao thức blockchain hay một kết quả thực nghiệm đã được hiệu chỉnh bằng dữ liệu thực.

## Các thành phần đang mô phỏng

- **Miner**: có capacity, unit cost, reward, profit và trạng thái `ACTIVE` hoặc `OFFLINE`.
- **Dataset**: có kích thước, số replica mục tiêu và danh sách miner đang lưu replica.
- **Bonding curve**: tính giá lưu trữ theo utilization của mạng.
- **Miner exit**: miner có xác suất rời mạng dựa trên profit.
- **Repair game**: khi dataset thiếu replica, miner còn capacity có thể nhận repair với xác suất phụ thuộc utility.
- **Endowment**: nhận yield cố định và chi trả storage reward cùng repair reward.
- **Cost decline**: unit cost của miner giảm theo tháng theo giả định `C_(t+1) = C_t × (1 − decline_rate)`.
- **Metrics**: ghi lại số miner hoạt động, capacity, utilization, giá lưu trữ, tình trạng replica, repair, availability và Endowment theo từng tháng.

## Công thức chính

### Utilization

```text
used_storage = Σ(dataset_size × replica_count)
utilization = used_storage / active_capacity
```

### Bonding curve

```text
price(ρ) = p_min + (p_max - p_min) / (1 + exp(-k × (ρ - ρ0)))
```

### Miner economics

```text
storage_reward_i = storage_price × miner_used_capacity_i
operating_cost_i = unit_cost_i × miner_used_capacity_i
profit_i = storage_reward_i - operating_cost_i
```

### Xác suất rời mạng và repair

```text
P_leave = 1 / (1 + exp(beta_leave × profit))
replica_deficit = max(0, target_replicas - current_replicas)
repair_reward = base_reward × (1 + alpha × replica_deficit / target_replicas)
P_repair = 1 / (1 + exp(-beta_repair × repair_utility))
```

Trong code hiện tại, quyết định rời mạng được so sánh bằng `rng.random() * 4 < P_leave`; do đó xác suất exit thực tế là `P_leave / 4`, không phải toàn bộ `P_leave`. Đây là chi tiết triển khai cần tính đến khi diễn giải kết quả.

### Giảm chi phí và cập nhật Endowment

```text
unit_cost_(t+1) = unit_cost_t × (1 - monthly_cost_decline)
endowment_(t+1) = endowment_t × (1 + yield_rate)
                    - storage_reward_t - repair_cost_t
```

## Cấu trúc project

```text
.
├── agent.py       # MinerAgent và các phép tính economics của miner
├── dataset.py     # Dataset, replica count, deficit và trạng thái dữ liệu
├── mechanisms.py  # Bonding curve, repair reward và repair probability
├── model.py       # StorageModel và vòng lặp mô phỏng theo tháng
├── config.py      # Seed và tham số mô phỏng
├── run.py         # Entry point để chạy mô phỏng
└── README.md
```

## Yêu cầu

- Python 3.10 trở lên
- Không cần cài thư viện bên ngoài cho phiên bản hiện tại

## Cách chạy

Mở terminal tại thư mục dự án và chạy:

```powershell
python run.py
```

Hoặc trên một số máy Windows:

```powershell
py run.py
```

Chương trình in số tháng chạy, seed, số miner hoạt động, data availability và Endowment ở tháng cuối. Các metrics từng tháng nằm trong `model.metrics` khi sử dụng model từ Python.

## Cấu hình

Các tham số mặc định hiện nằm trong `config.py`, gồm:

| Nhóm | Tham số mặc định |
|---|---|
| Simulation | 12 tháng, seed 46 |
| Miners | 100 miner; capacity 100–500; unit cost 0.1–0.2 |
| Cost decline | 3% mỗi tháng |
| Datasets | 10 dataset; size 10–50; mục tiêu 5 replica/dataset |
| Bonding curve | `p_min=0.5`, `p_max=2.0`, `rho0=0.75`, `k=15` |
| Miner behavior | `beta_leave=2.0` |
| Repair | `base_reward=0`, `alpha=0`, `beta_repair=0` |
| Endowment | Ban đầu 10,000; yield 0.8% mỗi tháng |

Có thể chỉnh giá trị trong `CONFIG` ở `config.py` rồi chạy lại. Giữ seed cố định để tái tạo cùng chuỗi lựa chọn ngẫu nhiên; đổi seed để tạo một realization khác.

## Trình tự một tháng

```text
Tính capacity và utilization
        ↓
Tính storage price bằng bonding curve
        ↓
Tính reward, operating cost và profit của miner
        ↓
Miner quyết định rời mạng; replica trên miner rời bị gỡ
        ↓
Phát hiện replica deficit và chạy repair game
        ↓
Cập nhật Endowment và metrics
        ↓
Giảm unit cost 3% cho tháng tiếp theo
```

## Phạm vi và giới hạn hiện tại

Phiên bản hiện tại chưa mô phỏng:

- UserAgent, truy vấn hoặc phát sinh dataset mới theo nhu cầu người dùng
- Miner mới tham gia mạng (JOIN)
- Token price, slippage hoặc chuyển token thực tế
- CIR yield, GBM storage cost hay các cú sốc ngẫu nhiên
- Retrieval/popularity reward, slashing, Risk Buffer hoặc Insurance Fund
- Monte Carlo nhiều lần chạy, xuất bảng CSV hoặc tạo biểu đồ

Các giá trị cấu hình hiện tại là tham số khởi đầu để phát triển và kiểm tra phần mềm, **không phải tham số đã được hiệu chỉnh thực nghiệm**. Cần kiểm tra độ nhạy và so sánh nhiều seed trước khi dùng kết quả để đưa ra kết luận nghiên cứu.

## Hướng phát triển

1. Kiểm thử các công thức và invariants của miner–dataset.
2. So sánh fixed repair reward với dynamic repair reward.
3. Thêm kịch bản forced miner exit và xuất metrics/biểu đồ.
4. Bổ sung miner JOIN và UserAgent/demand.
5. Mở rộng sang yếu tố stochastic và Monte Carlo sau khi phiên bản tối giản ổn định.
