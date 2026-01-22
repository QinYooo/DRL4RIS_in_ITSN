function [w_opt, Phi_opt, sum_rate] = optimize_ris_zf_SDR(h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S, W_sat, gamma_k, gamma_j, P_max, P_b, P_s, N_iter, sigma2, W_sat_,G_SAT_,H_SAT_,H_SAT_SU_)
    % 卫星-地面融合通信系统联合优化算法
    % 输入参数（实际信道）:
    %   h_k: 基站-用户直连信道 (K x N_t)
    %   h_j: 基站-卫星用户直连信道 (J x N_t)
    %   h_s_k: 卫星-基站用户直连信道 (K x N_s)
    %   h_s_j: 卫星-卫星用户直连信道 (J x N_s)
    %   h_k_r: 用户-RIS信道 (K x N)
    %   h_j_r: 卫星用户-RIS信道 (J x N)
    %   G_BS: RIS-基站信道 (N x N_t)
    %   G_S: RIS-卫星信道 (N x N_s)
    %   W_sat: 卫星波束成形矩阵 (N_s x J)
    %   gamma_k: 基站用户SINR阈值 (K x 1)
    %   gamma_j: 卫星用户SINR阈值 (J x 1)
    %   P_max: 基站最大发射功率
    %   N_iter: 最大迭代次数
    %   sigma2: 噪声功率
    %
    % 输出参数:
    %   w_opt: 优化后的波束成形矩阵 (N_t x K)
    %   Phi_opt: 优化后的RIS相位矩阵 (N x N) (对角矩阵)
    %   sum_rate: 每次迭代的和速率记录
    dbstop if error
    % 参数提取
    [K, N_t] = size(h_k);
    J = size(h_j, 1);
    N = size(G_BS, 1);
    N_s = size(W_sat, 1);

    % 初始化变量
    w = zeros(N_t, K);
    Phi = 9*diag(ones(N, 1)); % 初始化为单位矩阵(无反射)
    sum_rate = zeros(N_iter, 1);
    converged = false;

    % 迭代优化
    for iter = 1:N_iter
        % 1. 计算有效信道
        H_eff_k = zeros(K, N_t); % 基站到地面用户的有效信道
        H_eff_j = zeros(J, N_t); % 基站到卫星用户的有效信道
        H_sat_eff_k = zeros(K, N_s); % 卫星到地面用户的有效信道
        H_sat_eff_j = zeros(J, N_s); % 卫星到卫星用户的有效信道

        for k = 1:K
            % 基站到地面用户k的有效信道
            H_eff_k(k, :) = h_k(k, :) + h_k_r(k, :) * Phi * G_BS;
            % 卫星到地面用户k的有效信道
            H_sat_eff_k(k, :) = h_s_k(k, :) + h_k_r(k, :) * Phi * G_S;
        end

        for j = 1:J
            % 基站到卫星用户j的有效信道
            H_eff_j(j, :) = h_j(j, :) + h_j_r(j, :) * Phi * G_BS;
            % 卫星到卫星用户j的有效信道
            H_sat_eff_j(j, :) = h_s_j(j, :) + h_j_r(j, :) * Phi * G_S;
        end
        
        disp("Before Iteration: "+iter);
        % 基站用户速率
        for k = 1:K
            interference = sigma2;

            for i = 1:K

                if i ~= k
                    interference = interference + P_b * abs(H_eff_k(k, :) * w(:, i)) ^ 2;
                end

            end

            for j = 1:J
                interference = interference + P_s * abs(H_sat_eff_k(k, :) * W_sat(:, j)) ^ 2;
            end

            SINR_k = P_b * abs(H_eff_k(k, :) * w(:, k)) ^ 2 / interference;
            disp("BS UE SINR("+k + "): "+10 * log10(SINR_k) + " dB");
        end

        % 卫星用户速率
        for j = 1:J
            interference = sigma2;

            for k = 1:K
                interference = interference + P_b * abs(H_eff_j(j, :) * w(:, k)) ^ 2;
            end

            SINR_j = P_s * abs(H_sat_eff_j(j, :) * W_sat(:, j)) ^ 2 / interference;
            disp("BS UE SINR(SU): "+10 * log10(SINR_j) + " dB");
        end

        disp("After ZF:");








        % 2. ZF波束赋形
        H_combine = [H_eff_k; H_eff_j]; % 合并所有用户的有效信道

        if rank(H_combine) < K + J
            % 信道矩阵不满秩，添加正则化
            w = H_combine' * inv(H_combine * H_combine' +1e-6 * eye(K + J));
        else
            w = H_combine' * inv(H_combine * H_combine');
        end

        w = w(:, 1:K); % 只保留地面用户的波束成形向量

        % 3. 功率分配 (灌水法)
        p = zeros(K, 1);

        for k = 1:K
            % 计算地面用户k的SINR
            signal_power = P_b*abs(H_eff_k(k, :) * w(:, k)) ^ 2;

            interference_power = 0;
            % 来自其他地面用户的干扰
            for m = 1:K

                if m ~= k
                    interference_power = interference_power + P_b*abs(H_eff_k(k, :) * w(:, m)) ^ 2;
                end

            end

            % 来自卫星的干扰
            for j = 1:J
                interference_power = interference_power + P_s*abs(H_sat_eff_k(k, :) * W_sat(:, j)) ^ 2;
            end

            % 计算满足SINR阈值所需的功率
            p(k) = max(gamma_k * (interference_power + sigma2) / signal_power, 0);
        end

        % 功率约束处理
        total_power(iter) = norm(sqrt(p).* (w.'),'fro')^2;
        disp("BS POWER: "+total_power(iter));

        if total_power(iter) > P_max
            % 功率超出限制，重新分配
            p = water_filling_power_allocation(H_eff_k, w, gamma_k, P_max, sigma2, H_sat_eff_k, W_sat);
        end

        % 4. 计算和速率
        current_sum_rate = 0;
        gamma_k_iter = zeros(K,1);
        gamma_j_iter = zeros(J,1);
        for k = 1:K
            signal_power = P_b*p(k) * abs(H_eff_k(k, :) * w(:, k)) ^ 2;
            interference_power = sigma2;

            % 来自其他地面用户的干扰
            for m = 1:K

                if m ~= k
                    interference_power = interference_power + P_b * p(m) * abs(H_eff_k(k, :) * w(:, m)) ^ 2;
                end

            end

            % 来自卫星的干扰
            for j = 1:J
                interference_power = interference_power + P_s*abs(H_sat_eff_k(k, :) * W_sat(:, j)) ^ 2;
            end
            gamma_k_iter(k) = signal_power/interference_power;
            disp("BS UE("+k+") SINR: "+pow2db(gamma_k_iter(k))+" dB");
            current_sum_rate = current_sum_rate + log2(1 + signal_power / interference_power);
        end

        for j = 1:J
            signal_power = P_s*abs(H_sat_eff_j(j, :) * W_sat(:, j)) ^ 2;
            interference_power = sigma2;

            % 来自基站的干扰
            for k = 1:K
                interference_power = interference_power + P_b * p(k) * abs(H_eff_j(j, :) * w(:, k)) ^ 2;
            end
            gamma_j_iter(j) = signal_power/interference_power;
            disp("LEO UE("+k+") SINR: "+pow2db(gamma_j_iter(j))+" dB");
            current_sum_rate = current_sum_rate + log2(1 + signal_power / interference_power);
        end

        sum_rate(iter) = current_sum_rate;
        w = (sqrt(p) .* w.').'; % 更新波束成形矩阵

        % 5. 准备优化RIS所需要的参数
        A = zeros(K, K, N, N); % 存储A矩阵
        a_ = zeros(K, K, N);
        a = zeros(K, K);

        B = zeros(K, N, N); % 存储A矩阵
        b_ = zeros(K, N);
        b = zeros(K, 1);

        C = zeros(K, N, N); % 存储A矩阵
        c_ = zeros(K, N);
        c = zeros(K, 1);
        for k = 1:K
            for m = 1:K
                A(k, m, :, :) = (diag(9 * h_k_r(k, :)) * G_BS * (w(:, m) * w(:, m)') * G_BS' * diag(9 * h_k_r(k, :))');
                a_(k, m, :) = (diag(9 * h_k_r(k, :)) * G_BS * (w(:, m) * w(:, m)') * h_k(k, :)');
                a(k, m) = (w(:, m)' * (h_k(k, :)' * h_k(k, :)) * w(:, m));
            end
            B(k, :, :) = (diag(9 * h_k_r(k, :)) * G_S * (W_sat * W_sat') * G_S' * diag(9 * h_k_r(k, :))');
            b_(k, :) = (diag(9 * h_k_r(k, :)) * G_S * (W_sat * W_sat') * h_s_k(k, :)');
            b(k) = (W_sat' * (h_s_k(k, :)' * h_s_k(k, :)) * W_sat);

            C(k, :, :) = (diag(9 * h_j_r) * G_BS * (w(:, k) * w(:, k)') * G_BS' * diag(9 * h_j_r)');
            c_(k, :) = (diag(9 * h_j_r) * G_BS * (w(:, k) * w(:, k)') * h_j');
            c(k) = (w(:, k)' * (h_j' * h_j) * w(:, k));
        end
        D = (diag(9 * h_j_r) * G_S * (W_sat * W_sat') * G_S' * diag(9 * h_j_r)');
        d_ = (diag(9 * h_j_r) * G_S * (W_sat * W_sat') * h_s_j');
        d = (W_sat' * (h_s_j' * h_s_j) * W_sat);

        Rb = zeros(K,N+1,N+1);
        Rc = zeros(K,N+1,N+1);
        Rd = zeros(N+1,N+1);
        for k = 1:K
            for m = 1:K
                A_km = reshape(A(k, m, :, :), N, N);
                a_km = reshape(a_(k, m, :), N, 1);
                Ra(k, m, :, :) = [A_km, a_km; a_km', 0];
            end
            B_k = reshape(B(k, :, :), N, N);
            b_k = reshape(b_(k, :), N, 1);
            C_k = reshape(C(k, :, :), N, N);
            c_k = reshape(c_(k, :), N, 1);
            Rb(k, :, :) = [B_k, b_k; b_k', 0];
            Rc(k, :, :) = [C_k, c_k; c_k', 0];
        end
        Rd = [D, d_; d_', 0];

        % 6. 使用CVX求解RIS相位优化 (SDR方法)
        cvx_begin quiet
            cvx_precision high;
            variable alpha_t(K, 1);
            variable beta_t;
            variable V(N+1, N+1) hermitian;
            expression RAV(K,K);
            for k = 1:K
                for m = 1:K
                    RAV(k, m) = 1e8 * trace(reshape(Ra(k, m, :, :), N + 1, N + 1) * V);
                end
            end
            

            maximize sum(alpha_t) + 16*beta_t;
            subject to
                V == hermitian_semidefinite(N + 1);
                diag(V) == 1;
                alpha_t >= 0;
                beta_t >= 0;
                sum_C = 0;
                for k = 1:K
                    P_b*real(RAV(k,k)+1e8*a(k,k)) >= gamma_k_iter(k) * (real(P_b*(sum(RAV(k,[1:k-1,k+1:end]))+1e8*sum(a(k, [1:k-1,k+1:end]))) + P_s*(1e8 * trace(reshape(Rb(k, :, :), N + 1, N + 1) * V)+ 1e8*b(k)) ) + 1e8*sigma2 ) + alpha_t(k);
                    sum_C = sum_C + 1e8 * trace(reshape(Rc(k, :, :), N + 1, N + 1) * V) + 1e8*c(k);
                end
                P_s * real(1e8 * trace(Rd * V)+1e8*d) >= gamma_j_iter * (real(P_b*sum_C) + 1e8*sigma2)+ beta_t;
        cvx_end
        


        % 检查优化状态
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            % 特征值分解恢复相位
            % [U, D_] = eig(V); % 特征分解
            % [d_max, idx] = max(diag(D_));        % 找到最大特征值
            % v_approx = sqrt(d_max) * U(:, idx); % 主特征向量缩放
            % v = exp(1j * angle(v_approx / v_approx(end)));
            % Phi_new = 9*diag(v(1 : N));

            % 高斯随机化过程
            max_F = 0;
            max_v = 0;
            [U, Sigma] = eig(V);
            L = 1000;
            [d_max, idx] = max(diag(Sigma)); % 找到最大特征值
            v_approx = sqrt(d_max) * U(:, idx); % 主特征向量缩放
            l = 1;
            disp("最大特征值主成分占比："+d_max / sum(diag(Sigma)));
            while (l < L)
                r = sqrt(2) / 2 * (randn(N + 1, 1) + 1j * randn(N + 1, 1));
                v = v_approx + (1 - d_max / sum(diag(Sigma))) * U * Sigma ^ (0.5) * r;

                v = exp(1j * angle(v / v(end)))';
                F = 0;
                sum_partC = 0;

                for k = 1:K
                    sum_part = 0;
                    Ra_kk = reshape(Ra(k, k, :, :), N + 1, N + 1);
                    Rb_k = reshape(Rb(k, :, :), N + 1, N + 1);
                    Rc_k = reshape(Rc(k, :, :), N + 1, N + 1);
                    c_k = c(k);
                    sum_partC = v * Rc_k * v' + c_k + sum_partC;

                    for m = [1:k - 1 k + 1:K]
                        Ra_km = reshape(Ra(k, m, :, :), N + 1, N + 1);
                        a_km = a(k, m);
                        sum_part = sum_part + real(v * Ra_km * v' + (a_km));
                    end

                    a_kk = a(k, k);
                    b_k = b(k);
                    F = F + log2(1 + (P_b * abs(v * Ra_kk * v' + a_kk)) / (abs(P_b * sum_part + P_s * abs(v * Rb_k * v' + b_k) + sigma2)));
                end

                F = F + log2(1 + P_s * abs(v * Rd * v' + real(d)) / abs(P_b * sum_partC + sigma2));

                if (F > max_F)
                    max_v = v;
                    max_F = F;
                end

                l = l + 1;
            end

            v = max_v;
            phi = v(1:N).';
            Phi = 9 * diag(phi); % 更新相位矩阵

        else
            % 如果优化失败，使用梯度下降法更新
            fprintf('CVX求解失败');
            % 此处省略梯度下降法的实现
        end
        
        % 更新P2
        for k = 1:K
            % 基站到地面用户k的有效信道
            H_eff_k(k, :) = h_k(k, :) + h_k_r(k, :) * Phi * G_BS;
            % 卫星到地面用户k的有效信道
            H_sat_eff_k(k, :) = h_s_k(k, :) + h_k_r(k, :) * Phi * G_S;
        end

        for j = 1:J
            % 基站到卫星用户j的有效信道
            H_eff_j(j, :) = h_j(j, :) + h_j_r(j, :) * Phi * G_BS;
            % 卫星到卫星用户j的有效信道
            H_sat_eff_j(j, :) = h_s_j(j, :) + h_j_r(j, :) * Phi * G_S;
        end
        
        for j = 1:J
            signal_power = abs(H_sat_eff_j(j, :) * W_sat(:, j)) ^ 2;
            interference_power = sigma2;

            % 来自基站的干扰
            for k = 1:K
                interference_power = interference_power + P_b * abs(H_eff_j(j, :) * w(:, k)) ^ 2;
            end
            P_s = gamma_j*interference_power/signal_power;
        end
        disp("P_sum:"+(P_b*norm(w,'fro')^2+P_s*N_s));
        disp("P_bs:"+P_b*norm(w,'fro')^2);
        disp("P_sat:"+P_s*N_s);
        P_sum(iter+1)=P_b*norm(w,'fro')^2+P_s*N_s;


        % 检查收敛
        if abs(P_sum(iter+1)-P_sum(iter))<1e-2 || iter >= N_iter
            break;
        end

    end

    % 保存最优解
    w_opt = w;
    Phi_opt = Phi;
    
    % 绘制收敛曲线
    figure;
    plot(1:iter, sum_rate(1:iter), 'LineWidth', 2);
    xlabel('迭代次数');
    ylabel('和速率 (bps/Hz)');
    title('算法收敛曲线');
    grid on;
end

function p = water_filling_power_allocation(H, w, gamma, P_max, sigma2, H_sat, W_sat)
    % 灌水法功率分配
    % 输入:
    %   H: 有效信道矩阵 (K x N_t)
    %   w: 波束成形矩阵 (N_t x K)
    %   gamma: SINR阈值 (K x 1)
    %   P_max: 最大发射功率
    %   sigma2: 噪声功率
    %   H_sat: 卫星到地面用户的有效信道 (K x N_s)
    %   W_sat: 卫星波束成形矩阵 (N_s x J)
    % 输出:
    %   p: 功率分配向量 (K x 1)

    K = size(H, 1);
    mu = 1e-3; % 初始水水平面
    step = 1e-2; % 步长
    max_iter = 1000; % 最大迭代次数
    epsilon = 1e-6; % 收敛容差

    % 计算每个用户的信道增益
    g = zeros(K, 1);

    for k = 1:K
        g(k) = abs(H(k, :) * w(:, k)) ^ 2;

        % 计算干扰功率
        interference = sigma2;

        for j = 1:size(W_sat, 2)
            interference = interference + abs(H_sat(k, :) * W_sat(:, j)) ^ 2;
        end

        g(k) = g(k) / (gamma(k) * interference);
    end

    % 排序
    [g_sorted, idx] = sort(g, 'descend');
    idx_map = zeros(K, 1);
    idx_map(idx) = 1:K;

    % 灌水法迭代
    for iter = 1:max_iter
        p = zeros(K, 1);
        active_users = 0;

        % 计算当前水水平面下的活跃用户
        for i = 1:K

            if 1 / g_sorted(i) < mu
                active_users = active_users + 1;
            else
                break;
            end

        end

        if active_users > 0
            % 计算活跃用户的功率
            for i = 1:active_users
                p(idx_map(idx(i))) = mu - 1 / g_sorted(i);
            end

            % 计算总功率
            total_power = sum(p);

            % 调整水水平面
            if abs(total_power - P_max) < epsilon
                break;
            elseif total_power > P_max
                mu = mu - step;
                step = step / 2;
            else
                mu = mu + step;
            end

        else
            % 如果没有活跃用户，增加水水平面
            mu = mu + step;
        end

        if iter >= max_iter
            fprintf('灌水法未收敛\n');
            break;
        end

    end

    % 确保功率非负
    p = max(p, 0);
end
