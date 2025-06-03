import random
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import matplotlib.font_manager as fm
import matplotlib
from pylab import mpl
import os

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

STRATEGY_NAMES = {
    "Honest": "诚实者",
    "Repentant": "悔改者",
    "KnightErrant": "游侠",
    "Detective": "侦探",
    "Poacher": "偷猎者",
    "OrderAgent": "秩序卫士",
    "Zen": "禅师",
    "Custom": "节奏者",
    "TitForTat": "针锋相对",
    "Grudger": "记仇者",
    "Random": "随机者",
    "Pavlov": "巴甫洛夫",
    "Defector": "背叛者"
}

class Agent:
    def __init__(self, id, strategy, is_player=False, generation=1, coins=0):
        self.id = id
        self.strategy = strategy
        self.coins = coins
        self.infamy = 0
        self.is_revenger = False
        self.history = {}
        self.is_player = is_player
        self.generation = generation

    def decide_action(self, other, round_num):
        return self.strategy.decide(self, other, round_num)

    def update(self, action, other_action, other_id, round_num, coop_reward, defect_reward, both_defect_penalty, coop_loss):
        if action == "合作" and other_action == "合作":
            self.coins += coop_reward
            if self.is_revenger and random.random() < 0.3:
                self.is_revenger = False
                self.strategy = HonestStrategy()
        elif action == "合作" and other_action == "背叛":
            self.coins = max(0, self.coins - coop_loss)
            if self.strategy.name == STRATEGY_NAMES["Honest"] and random.random() < 0.1:
                self.is_revenger = True
                self.strategy = DefectorStrategy()
        elif action == "背叛" and other_action == "合作":
            self.coins += defect_reward
            self.infamy += 1
        elif action == "背叛" and other_action == "背叛":
            self.coins += both_defect_penalty
        if other_id not in self.history:
            self.history[other_id] = []
        self.history[other_id].append(other_action)

    def can_reproduce(self, threshold, newborn_coins=0):
        repro_cost = self.get_repro_cost(threshold)
        return self.coins >= (repro_cost + newborn_coins)

    def reproduce(self, cost, newborn_coins=0):
        repro_cost = self.get_repro_cost(cost)
        if self.coins >= (repro_cost + newborn_coins):
            self.coins -= (repro_cost + newborn_coins)
            return Agent(None, self.strategy, self.is_player, generation=self.generation + 1, coins=newborn_coins)
        return None

    def get_repro_cost(self, base_cost):
        recent_actions = []
        for other_id, actions in self.history.items():
            recent_actions.extend(actions[-10:])
        if recent_actions:
            coop_rate = sum(1 for a in recent_actions if a == "合作") / len(recent_actions)
            if coop_rate > 0.7:
                return base_cost * 0.8
        return base_cost

class Strategy:
    def __init__(self, name):
        self.name = name

    def decide(self, agent, other, round_num):
        raise NotImplementedError

class HonestStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Honest"])
    def decide(self, agent, other, round_num):
        return "背叛" if agent.is_revenger else "合作"

class RepentantStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Repentant"])
    def decide(self, agent, other, round_num):
        return agent.history[other.id][-1] if other.id in agent.history and agent.history[other.id] else "合作"

class KnightErrantStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["KnightErrant"])
    def decide(self, agent, other, round_num):
        return random.choice(["合作", "背叛"])

class DetectiveStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Detective"])
    def decide(self, agent, other, round_num):
        if round_num <= 3 or other.id not in agent.history or not agent.history[other.id]:
            return "背叛"
        deception_rate = sum(1 for a in agent.history[other.id] if a == "背叛") / len(agent.history[other.id])
        return "背叛" if deception_rate <= 0.4 else "合作"

class PoacherStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Poacher"])
    def decide(self, agent, other, round_num):
        if round_num <= 3 or other.id not in agent.history or not agent.history[other.id]:
            return "合作"
        deception_rate = sum(1 for a in agent.history[other.id] if a == "背叛") / len(agent.history[other.id])
        return "背叛" if deception_rate > 0.4 else "合作"

class OrderAgentStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["OrderAgent"])
    def decide(self, agent, other, round_num):
        return "背叛" if other.infamy >= 2 else "合作"

class ZenAgentStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Zen"])
    def decide(self, agent, other, round_num):
        return "合作"

class CustomStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Custom"])
    def decide(self, agent, other, round_num):
        return "合作" if round_num % 2 == 0 else "背叛"

class TitForTatStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["TitForTat"])
    def decide(self, agent, other, round_num):
        return "合作" if other.id not in agent.history or not agent.history[other.id] else agent.history[other.id][-1]

class GrudgerStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Grudger"])
    def decide(self, agent, other, round_num):
        return "背叛" if other.id in agent.history and "背叛" in agent.history[other.id] else "合作"

class RandomStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Random"])
    def decide(self, agent, other, round_num):
        return random.choice(["合作", "背叛"])

class PavlovStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Pavlov"])
    def decide(self, agent, other, round_num):
        if other.id not in agent.history or len(agent.history[other.id]) < 2:
            return "合作"
        last_action, last_result = agent.history[other.id][-1], agent.history[other.id][-2]
        if last_action == "合作" and last_result == "合作":
            return "合作"
        elif last_action == "背叛" and last_result == "背叛":
            return "背叛"
        return "背叛" if last_action == "合作" else "合作"

class DefectorStrategy(Strategy):
    def __init__(self):
        super().__init__(STRATEGY_NAMES["Defector"])
    def decide(self, agent, other, round_num):
        return "背叛"

class FloatingIsland:
    def __init__(self, base_production, capacity=10, disaster_prob=0.1, boom_prob=0.1):
        self.base_production = base_production
        self.temporary_effects = []
        self.capacity = capacity
        self.disaster_prob = disaster_prob
        self.boom_prob = boom_prob

    def apply_event(self):
        if random.random() < self.disaster_prob:
            self.capacity = max(1, self.capacity - random.randint(1, 3))
            print(f"灾难！岛屿容量减少至 {self.capacity}")
        elif random.random() < self.boom_prob:
            self.capacity += random.randint(1, 3)
            print(f"繁荣！岛屿容量增加至 {self.capacity}")

class Game:
    def __init__(self, mode="standard", level_name=None, strategy_counts=None, player_strategy_counts=None, island_capacity=10, disaster_prob=0.1, boom_prob=0.1,
                 coop_reward=10, defect_reward=15, both_defect_penalty=5, repro_threshold=100, repro_cost=50, coop_loss=0, base_production=100, newborn_coins=10):
        self.mode = mode
        self.level_name = level_name
        self.island = FloatingIsland(base_production=base_production, capacity=island_capacity, disaster_prob=disaster_prob, boom_prob=boom_prob)
        self.agents = []
        self.round = 0
        self.total_coins = 0
        self.round_choices = []
        self.next_id = 0
        self.coop_reward = coop_reward
        self.defect_reward = defect_reward
        self.both_defect_penalty = both_defect_penalty
        self.repro_threshold = repro_threshold
        self.repro_cost = repro_cost
        self.coop_loss = coop_loss
        self.newborn_coins = newborn_coins
        self.strategy_counts_history = []
        self.coop_defect_history = []
        self.player_machine_counts_history = []
        self.strategy_generations = defaultdict(list)

        strategy_map = {
            STRATEGY_NAMES["Honest"]: HonestStrategy,
            STRATEGY_NAMES["Repentant"]: RepentantStrategy,
            STRATEGY_NAMES["KnightErrant"]: KnightErrantStrategy,
            STRATEGY_NAMES["Detective"]: DetectiveStrategy,
            STRATEGY_NAMES["Poacher"]: PoacherStrategy,
            STRATEGY_NAMES["OrderAgent"]: OrderAgentStrategy,
            STRATEGY_NAMES["Zen"]: ZenAgentStrategy,
            STRATEGY_NAMES["Custom"]: CustomStrategy,
            STRATEGY_NAMES["TitForTat"]: TitForTatStrategy,
            STRATEGY_NAMES["Grudger"]: GrudgerStrategy,
            STRATEGY_NAMES["Random"]: RandomStrategy,
            STRATEGY_NAMES["Pavlov"]: PavlovStrategy,
            STRATEGY_NAMES["Defector"]: DefectorStrategy
        }

        if self.mode == "brawl":
            strategies = list(strategy_map.keys())
            num_strategies = len(strategies)
            agents_per_strategy = island_capacity // num_strategies
            for strategy_name in strategies:
                strategy = strategy_map[strategy_name]()
                for _ in range(agents_per_strategy):
                    agent = Agent(self.next_id, strategy, is_player=True)
                    self.agents.append(agent)
                    self.next_id += 1
        else:
            for strategy_name, count in strategy_counts.items():
                if strategy_name in strategy_map:
                    for _ in range(count):
                        strategy = strategy_map[strategy_name]()
                        agent = Agent(self.next_id, strategy, is_player=False)
                        self.agents.append(agent)
                        self.next_id += 1
            for strategy_name, count in player_strategy_counts.items():
                if strategy_name in strategy_map:
                    for _ in range(count):
                        strategy = strategy_map[strategy_name]()
                        agent = Agent(self.next_id, strategy, is_player=True)
                        self.agents.append(agent)
                        self.next_id += 1

    def distribute_coins(self):
        current_temp_multiplier = 1.0
        for effect in self.island.temporary_effects:
            if effect['start'] <= self.round <= effect['end']:
                current_temp_multiplier *= effect['multiplier']
        
        num_agents = len(self.agents)
        if num_agents > 0:
            population_multiplier = self.island.capacity / max(1, num_agents)
        else:
            population_multiplier = 1.0

        daily_production = self.island.base_production * current_temp_multiplier * population_multiplier
        self.total_coins += daily_production
        if self.agents:
            per_agent = int(daily_production // num_agents)
            for agent in self.agents:
                agent.coins += per_agent

    def run_interactions(self):
        agents_copy = self.agents.copy()
        random.shuffle(agents_copy)
        round_choices = []
        coop_count = 0
        defect_count = 0
        for i in range(0, len(agents_copy), 2):
            if i + 1 < len(agents_copy):
                a1, a2 = agents_copy[i], agents_copy[i + 1]
                action1 = a1.decide_action(a2, self.round)
                action2 = a2.decide_action(a1, self.round)
                round_choices.append((a1.id, action1))
                round_choices.append((a2.id, action2))
                a1.update(action1, action2, a2.id, self.round, self.coop_reward, self.defect_reward, self.both_defect_penalty, self.coop_loss)
                a2.update(action2, action1, a1.id, self.round, self.coop_reward, self.defect_reward, self.both_defect_penalty, self.coop_loss)
                coop_count += (action1 == "合作") + (action2 == "合作")
                defect_count += (action1 == "背叛") + (action2 == "背叛")
        self.round_choices.append(round_choices)
        total_actions = coop_count + defect_count
        coop_rate = coop_count / total_actions if total_actions > 0 else 0
        defect_rate = defect_count / total_actions if total_actions > 0 else 0
        self.coop_defect_history.append((coop_rate, defect_rate))

    def record_strategy_counts(self):
        strategy_count = defaultdict(int)
        player_count = 0
        machine_count = 0
        for agent in self.agents:
            strategy_count[agent.strategy.name] += 1
            if agent.is_player:
                player_count += 1
            else:
                machine_count += 1
        self.strategy_counts_history.append(dict(strategy_count))
        self.player_machine_counts_history.append({"玩家": player_count, "机器": machine_count})

    def apply_environmental_effects(self):
        if self.round % 10 == 0 and len(self.round_choices) >= 10:
            last_10_choices = self.round_choices[-10:]
            total_C = sum(sum(1 for _, action in choices if action == "合作") for choices in last_10_choices)
            total_D = sum(sum(1 for _, action in choices if action == "背叛") for choices in last_10_choices)
            total_N = total_C + total_D
            if total_N > 0:
                avg_coop = total_C / total_N
                avg_decep = total_D / total_N
                if avg_coop >= 0.7:
                    self.island.base_production *= 1.5
                if avg_decep >= 0.6:
                    self.island.temporary_effects.append({'start': self.round + 1, 'end': self.round + 10, 'multiplier': 0.7})
            if random.random() < 0.3:
                self.island.temporary_effects.append({'start': self.round + 1, 'end': self.round + 2, 'multiplier': 0.5})

    def reproduce_agents(self):
        new_agents = []
        for agent in self.agents:
            if agent.can_reproduce(self.repro_threshold, self.newborn_coins):
                new_agent = agent.reproduce(self.repro_cost, self.newborn_coins)
                if new_agent:
                    new_agent.id = self.next_id
                    self.next_id += 1
                    new_agents.append(new_agent)
                    self.strategy_generations[agent.strategy.name].append(new_agent.generation)
        self.agents.extend(new_agents)

    def eliminate_agents(self):
        while len(self.agents) > self.island.capacity:
            if not self.agents:
                break
            min_coins_agent = min(self.agents, key=lambda a: a.coins)
            self.agents.remove(min_coins_agent)
            print(f"代理 {min_coins_agent.id} 被淘汰（硬币不足）")

    def run_round(self):
        self.round += 1
        self.distribute_coins()
        self.run_interactions()
        self.record_strategy_counts()
        self.apply_environmental_effects()
        self.reproduce_agents()
        self.eliminate_agents()
        self.island.apply_event()

    def run_simulation(self, num_rounds):
        if self.level_name:
            output_dir = f"graphs/{self.level_name}"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "graphs"
            os.makedirs(output_dir, exist_ok=True)

        for _ in range(num_rounds):
            self.run_round()
            print(f"轮次 {self.round}: 代理总数 = {len(self.agents)}, 岛屿容量 = {self.island.capacity}")
        self.plot_strategy_counts(output_dir)
        self.plot_coop_defect_rates(output_dir)
        self.plot_player_machine_counts(output_dir)
        self.plot_generation_stats(output_dir)

    def plot_strategy_counts(self, output_dir):
        rounds = list(range(1, self.round + 1))
        strategies = set().union(*(d.keys() for d in self.strategy_counts_history))
        plt.figure(figsize=(12, 8))
        for strategy in strategies:
            counts = [count_dict.get(strategy, 0) for count_dict in self.strategy_counts_history]
            line, = plt.plot(rounds, counts, label=strategy)
            if counts[-1] > 0:
                x = rounds[-1]
                y = counts[-1]
                offset_y = (list(strategies).index(strategy) % 4 - 2) * 0.5
                plt.text(x + 5, y + offset_y, strategy, fontsize=8, ha='left', va='center', color=line.get_color())
        plt.xlabel("轮次")
        plt.ylabel("代理数量")
        plt.title("各策略代理数量随轮次变化")
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"{output_dir}/strategy_counts_{timestamp}.png", bbox_inches='tight')
        plt.close()

    def plot_coop_defect_rates(self, output_dir):
        rounds = list(range(1, self.round + 1))
        coop_rates = [data[0] for data in self.coop_defect_history]
        defect_rates = [data[1] for data in self.coop_defect_history]
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, coop_rates, label="合作率", color="green")
        plt.plot(rounds, defect_rates, label="背叛率", color="red")
        plt.xlabel("轮次")
        plt.ylabel("比率")
        plt.title("合作与背叛率随轮次变化")
        plt.legend()
        plt.grid(True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"{output_dir}/coop_defect_rates_{timestamp}.png")
        plt.close()

    def plot_player_machine_counts(self, output_dir):
        rounds = list(range(1, self.round + 1))
        player_counts = [data["玩家"] for data in self.player_machine_counts_history]
        machine_counts = [data["机器"] for data in self.player_machine_counts_history]
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, player_counts, label="玩家代理", color="blue")
        if self.mode == "standard":
            plt.plot(rounds, machine_counts, label="机器代理", color="orange")
        plt.xlabel("轮次")
        plt.ylabel("代理数量")
        plt.title("代理数量随轮次变化")
        plt.legend()
        plt.grid(True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"{output_dir}/player_machine_counts_{timestamp}.png")
        plt.close()

    def plot_generation_stats(self, output_dir):
        strategies = list(self.strategy_generations.keys())
        avg_generations = []
        max_generations = []
        for strategy in strategies:
            generations = self.strategy_generations[strategy]
            if generations:
                avg_generations.append(sum(generations) / len(generations))
                max_generations.append(max(generations))
            else:
                avg_generations.append(0)
                max_generations.append(0)
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.35
        index = range(len(strategies))
        ax.bar(index, avg_generations, bar_width, label='平均代数', color='blue')
        ax.bar([i + bar_width for i in index], max_generations, bar_width, label='最大代数', color='orange')
        ax.set_xlabel('策略')
        ax.set_ylabel('代数')
        ax.set_title('各策略代数统计')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10)
        ax.legend()
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"{output_dir}/generation_stats_{timestamp}.png")
        plt.close()

LEVELS = [
    {
        "name": "新手信任",
        "description": "你降落在一个阳光普照的浮岛，资源丰富，居民友善。建立繁荣社区，依靠合作立足！",
        "coop_reward": 20,
        "defect_reward": 25,
        "coop_loss": 10,
        "both_defect_penalty": -5,
        "repro_threshold": 60,
        "repro_cost": 15,
        "island_capacity": 200,
        "disaster_prob": 0.03,
        "boom_prob": 0.30,
        "base_production": 100,
        "newborn_coins": 20
    },
    {
        "name": "随机混战",
        "description": "喧嚣的浮岛集市，商人流浪者行为不定。应对随机选择，争夺资源！",
        "coop_reward": 15,
        "defect_reward": 20,
        "coop_loss": 12,
        "both_defect_penalty": -8,
        "repro_threshold": 90,
        "repro_cost": 25,
        "island_capacity": 200,
        "disaster_prob": 0.08,
        "boom_prob": 0.20,
        "base_production": 120,
        "newborn_coins": 20
    },
    {
        "name": "针锋相对",
        "description": "云雾缭绕的浮岛，居民遵循互惠法则。精准应对对手行动，占据一席之地！",
        "coop_reward": 15,
        "defect_reward": 18,
        "coop_loss": 10,
        "both_defect_penalty": -8,
        "repro_threshold": 120,
        "repro_cost": 30,
        "island_capacity": 150,
        "disaster_prob": 0.10,
        "boom_prob": 0.15,
        "base_production": 90,
        "newborn_coins": 20
    },
    {
        "name": "适应竞技场",
        "description": "风暴摧残的浮岛，资源稀缺。智胜对手，抵御灾难，繁荣发展！",
        "coop_reward": 12,
        "defect_reward": 18,
        "coop_loss": 15,
        "both_defect_penalty": -10,
        "repro_threshold": 150,
        "repro_cost": 40,
        "island_capacity": 120,
        "disaster_prob": 0.10,
        "boom_prob": 0.08,
        "base_production": 80,
        "newborn_coins": 25
    },
    {
        "name": "终极生存",
        "description": "荒凉浮岛，资源枯竭。背叛回报高，环境惩罚严酷，唯有精明策略称霸！",
        "coop_reward": 12,
        "defect_reward": 20,
        "coop_loss": 18,
        "both_defect_penalty": -12,
        "repro_threshold": 180,
        "repro_cost": 45,
        "island_capacity": 100,
        "disaster_prob": 0.15,
        "boom_prob": 0.08,
        "base_production": 70,
        "newborn_coins": 30
    },
    {
        "name": "进化生态",
        "description": "生机勃勃的浮岛，合作与竞争平衡。快速适应，统治动态乐园！",
        "coop_reward": 14,
        "defect_reward": 18,
        "coop_loss": 12,
        "both_defect_penalty": -8,
        "repro_threshold": 135,
        "repro_cost": 35,
        "island_capacity": 140,
        "disaster_prob": 0.12,
        "boom_prob": 0.15,
        "base_production": 95,
        "newborn_coins": 20
    },
    {
        "name": "混乱熔炉",
        "description": "动荡不安的浮岛，投机者与策略家争夺控制权。敏锐果断，夺取权力！",
        "coop_reward": 12,
        "defect_reward": 18,
        "coop_loss": 15,
        "both_defect_penalty": -10,
        "repro_threshold": 165,
        "repro_cost": 40,
        "island_capacity": 100,
        "disaster_prob": 0.15,
        "boom_prob": 0.10,
        "base_production": 75,
        "newborn_coins": 25
    },
    {
        "name": "资源争夺战",
        "description": "矿藏丰富的浮岛，争夺激烈。平衡合作与背叛，守护财富！",
        "coop_reward": 15,
        "defect_reward": 20,
        "coop_loss": 12,
        "both_defect_penalty": -8,
        "repro_threshold": 120,
        "repro_cost": 30,
        "island_capacity": 160,
        "disaster_prob": 0.10,
        "boom_prob": 0.12,
        "base_production": 110,
        "newborn_coins": 20
    },
    {
        "name": "战略迷雾",
        "description": "神秘迷雾笼罩的浮岛，居民伪装试探。洞察意图，制定完美战略！",
        "coop_reward": 13,
        "defect_reward": 18,
        "coop_loss": 14,
        "both_defect_penalty": -9,
        "repro_threshold": 140,
        "repro_cost": 35,
        "island_capacity": 120,
        "disaster_prob": 0.15,
        "boom_prob": 0.10,
        "base_production": 85,
        "newborn_coins": 25
    },
    {
        "name": "末日试炼",
        "description": "濒临毁灭的浮岛，资源匮乏。背叛常态，坚韧策略坚持到最后！",
        "coop_reward": 15,
        "defect_reward": 20,
        "coop_loss": 20,
        "both_defect_penalty": -10,
        "repro_threshold": 200,
        "repro_cost": 50,
        "island_capacity": 80,
        "disaster_prob": 0.18,
        "boom_prob": 0.10,
        "base_production": 60,
        "newborn_coins": 30
    },
    {
        "name": "均衡博弈",
        "description": "平静云海中的浮岛，资源均匀。精于博弈，找到制胜之道！",
        "coop_reward": 15,
        "defect_reward": 18,
        "coop_loss": 10,
        "both_defect_penalty": -8,
        "repro_threshold": 120,
        "repro_cost": 30,
        "island_capacity": 160,
        "disaster_prob": 0.10,
        "boom_prob": 0.15,
        "base_production": 100,
        "newborn_coins": 20
    },
    {
        "name": "绝境反击",
        "description": "雷暴包围的浮岛，资源稀缺。背叛看似唯一出路，合作或带来转机！",
        "coop_reward": 12,
        "defect_reward": 20,
        "coop_loss": 18,
        "both_defect_penalty": -10,
        "repro_threshold": 150,
        "repro_cost": 40,
        "island_capacity": 160,
        "disaster_prob": 0.18,
        "boom_prob": 0.08,
        "base_production": 65,
        "newborn_coins": 25
    }
]

def choose_machine_strategies(level, level_index):
    total_agents = level["island_capacity"] // 2
    strategy_counts = defaultdict(int)
    strategies = list(STRATEGY_NAMES.values())

    if level_index == 0:
        for _ in range(total_agents):
            strategy_counts[STRATEGY_NAMES["Zen"]] += 1
    elif level_index == 1:
        for _ in range(total_agents):
            strategy_counts[random.choice([STRATEGY_NAMES["Zen"], STRATEGY_NAMES["Defector"], STRATEGY_NAMES["KnightErrant"]])] += 1
    elif level_index == 2:
        reciprocal = [STRATEGY_NAMES["TitForTat"], STRATEGY_NAMES["Repentant"]]
        for _ in range(total_agents // 2):
            strategy_counts[random.choice(reciprocal)] += 1
        for _ in range(total_agents - sum(strategy_counts.values())):
            strategy_counts[random.choice([STRATEGY_NAMES["Random"], STRATEGY_NAMES["Honest"]])] += 1
    elif level_index == 3:
        adaptive = [STRATEGY_NAMES["Detective"], STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["Pavlov"]]
        defecting = [STRATEGY_NAMES["Defector"], STRATEGY_NAMES["Grudger"]]
        for _ in range(total_agents // 2):
            strategy_counts[random.choice(adaptive)] += 1
        for _ in range(total_agents // 4):
            strategy_counts[random.choice(defecting)] += 1
        for _ in range(total_agents - sum(strategy_counts.values())):
            strategy_counts[random.choice(strategies)] += 1
    elif level_index == 4:
        complex_strategies = [STRATEGY_NAMES["Detective"], STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["Pavlov"], STRATEGY_NAMES["OrderAgent"], STRATEGY_NAMES["Grudger"]]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        for _ in range(total_agents):
            strategy_counts[random.choices(complex_strategies, weights=weights, k=1)[0]] += 1
    elif level_index == 5:
        adaptive = [STRATEGY_NAMES["Pavlov"], STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["Detective"], STRATEGY_NAMES["OrderAgent"]]
        for _ in range(total_agents // 2):
            strategy_counts[random.choice(adaptive)] += 1
        for _ in range(total_agents - sum(strategy_counts.values())):
            strategy_counts[random.choice([STRATEGY_NAMES["TitForTat"], STRATEGY_NAMES["Grudger"], STRATEGY_NAMES["Random"]])] += 1
    elif level_index == 6:
        complex_strategies = [STRATEGY_NAMES["Detective"], STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["Pavlov"], STRATEGY_NAMES["Grudger"], STRATEGY_NAMES["Custom"]]
        weights = [0.25, 0.25, 0.2, 0.15, 0.15] if level["defect_reward"] > level["coop_reward"] else [0.2, 0.2, 0.3, 0.15, 0.15]
        for _ in range(total_agents):
            strategy_counts[random.choices(complex_strategies, weights=weights, k=1)[0]] += 1
    elif level_index == 7:
        mixed = [STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["Defector"], STRATEGY_NAMES["TitForTat"]]
        for _ in range(total_agents // 2):
            strategy_counts[random.choice(mixed)] += 1
        for _ in range(total_agents - sum(strategy_counts.values())):
            strategy_counts[random.choice([STRATEGY_NAMES["Random"], STRATEGY_NAMES["Honest"], STRATEGY_NAMES["Pavlov"]])] += 1
    elif level_index == 8:
        probing = [STRATEGY_NAMES["Detective"], STRATEGY_NAMES["Pavlov"], STRATEGY_NAMES["Custom"]]
        for _ in range(total_agents // 2):
            strategy_counts[random.choice(probing)] += 1
        for _ in range(total_agents - sum(strategy_counts.values())):
            strategy_counts[random.choice([STRATEGY_NAMES["Grudger"], STRATEGY_NAMES["OrderAgent"], STRATEGY_NAMES["Random"]])] += 1
    elif level_index == 9:
        complex_strategies = [STRATEGY_NAMES["Defector"], STRATEGY_NAMES["Detective"], STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["Grudger"]]
        weights = [0.4, 0.2, 0.2, 0.2]
        for _ in range(total_agents):
            strategy_counts[random.choices(complex_strategies, weights=weights, k=1)[0]] += 1
    elif level_index == 10:
        balanced = [STRATEGY_NAMES["TitForTat"], STRATEGY_NAMES["Pavlov"], STRATEGY_NAMES["Detective"]]
        for _ in range(total_agents // 2):
            strategy_counts[random.choice(balanced)] += 1
        for _ in range(total_agents - sum(strategy_counts.values())):
            strategy_counts[random.choice([STRATEGY_NAMES["Honest"], STRATEGY_NAMES["Random"], STRATEGY_NAMES["Grudger"]])] += 1
    elif level_index == 11:
        complex_strategies = [STRATEGY_NAMES["Defector"], STRATEGY_NAMES["Poacher"], STRATEGY_NAMES["OrderAgent"], STRATEGY_NAMES["Pavlov"]]
        weights = [0.35, 0.25, 0.2, 0.2]
        for _ in range(total_agents):
            strategy_counts[random.choices(complex_strategies, weights=weights, k=1)[0]] += 1

    return strategy_counts

def display_role_descriptions():
    roles = [
        (STRATEGY_NAMES["Honest"], "一位值得信赖的岛民，总是选择合作，但若被背叛（10%概率），会化为复仇者，永远背叛，直到双方合作（30%概率恢复诚实）。适合信任环境，但需防背叛。"),
        (STRATEGY_NAMES["Repentant"], "一位宽容的策略家，模仿对手上次的行动，首次合作。若对手背叛，他会报复，但若对手合作，他也会原谅。适合与互惠者共存。"),
        (STRATEGY_NAMES["KnightErrant"], "一位随性的冒险者，随机选择合作或背叛，行为不可预测。他在混乱环境中可能占优，但缺乏长期规划。"),
        (STRATEGY_NAMES["Detective"], "一位谨慎的观察者，前三轮或无历史时背叛，之后根据对手背叛率（≤40%背叛，>40%合作）决定行动。适合试探对手意图。"),
        (STRATEGY_NAMES["Poacher"], "一位狡猾的猎人，前三轮或无历史时合作，之后若对手背叛率>40%则背叛，否则合作。适合应对背叛者。"),
        (STRATEGY_NAMES["OrderAgent"], "一位正义的执法者，对恶名≥2的对手背叛，否则合作。他惩罚臭名昭著者，适合高恶名环境。"),
        (STRATEGY_NAMES["Zen"], "一位平和的智者，始终选择合作，无视对手行为。他在合作环境中如鱼得水，但在背叛盛行时易受损。"),
        (STRATEGY_NAMES["Custom"], "一位按时间行事的策略家，偶数轮合作，奇数轮背叛。他节奏固定，适合短期博弈但易被预测。"),
        (STRATEGY_NAMES["TitForTat"], "一位公平的交易者，首次合作，之后模仿对手上次的行动。他奖励合作，惩罚背叛，适合互惠环境。"),
        (STRATEGY_NAMES["Grudger"], "一位绝不宽恕的战士，首次合作，但若对手曾背叛，则永远背叛。他适合惩罚背叛者，但难以原谅。"),
        (STRATEGY_NAMES["Random"], "一位毫无规律的岛民，随机选择合作或背叛。他在混乱环境中可能占优，但无明确目标。"),
        (STRATEGY_NAMES["Pavlov"], "一位学习者，首次合作，之后根据上次行动和结果调整：合作成功继续合作，背叛成功继续背叛，否则切换。他适应动态环境。"),
        (STRATEGY_NAMES["Defector"], "一位自私的掠夺者，始终选择背叛，追求即时利益。他在背叛获利高的环境中占优，但易引发报复。")
    ]
    print("\n=== 角色说明 ===")
    for name, desc in roles:
        print(f"{name}：{desc}")
    print("================\n")

def main():
    display_role_descriptions()
    print("选择游戏模式：")
    print("1. 标准模式")
    print("2. 大乱斗模式")
    mode_choice = int(input("请输入模式编号 (1-2)："))
    mode = "brawl" if mode_choice == 2 else "standard"

    print("可用关卡：")
    for i, level in enumerate(LEVELS):
        print(f"{i + 1}. {level['name']}：{level['description']}")
    level_choice = int(input("选择一个关卡 (1-12)：")) - 1
    if level_choice not in range(len(LEVELS)):
        print("无效关卡，默认选择针锋相对")
        level_choice = 2
    level = LEVELS[level_choice]
    print(f"\n正在玩关卡：{level['name']}")
    print(f"描述：{level['description']}")
    print(f"条件：合作奖励={level['coop_reward']}，背叛奖励={level['defect_reward']}，"
          f"合作损失={level['coop_loss']}，双背叛惩罚={level['both_defect_penalty']}，"
          f"繁殖门槛={level['repro_threshold']}，繁殖成本={level['repro_cost']}，"
          f"岛屿容量={level['island_capacity']}，灾难概率={level['disaster_prob']}，"
          f"繁荣概率={level['boom_prob']}，基础生产力={level['base_production']}，"
          f"新生代初始金币={level['newborn_coins']}")

    if mode == "standard":
        machine_strategy_counts = choose_machine_strategies(level, level_choice)
        print("\n机器的策略分配：")
        for strategy, count in machine_strategy_counts.items():
            if count > 0:
                print(f"{strategy}: {count}")
        strategies = list(STRATEGY_NAMES.values())
        player_strategy_counts = {}
        max_player_agents = level["island_capacity"] // 2
        print(f"\n为每个策略输入代理数量（总数不得超过 {max_player_agents}）：")
        total_player_agents = 0
        for strategy in strategies:
            count = int(input(f"{strategy}："))
            if count < 0:
                count = 0
            player_strategy_counts[strategy] = count
            total_player_agents += count
            if total_player_agents > max_player_agents:
                print(f"警告：代理总数 ({total_player_agents}) 超过限制 ({max_player_agents})，自动截断。")
                player_strategy_counts[strategy] -= (total_player_agents - max_player_agents)
                total_player_agents = max_player_agents
                break
        print("\n玩家的策略分配：")
        for strategy, count in player_strategy_counts.items():
            if count > 0:
                print(f"{strategy}: {count}")
        game = Game(mode=mode, level_name=level["name"], strategy_counts=machine_strategy_counts, player_strategy_counts=player_strategy_counts,
                    island_capacity=level["island_capacity"], disaster_prob=level["disaster_prob"],
                    boom_prob=level["boom_prob"], coop_reward=level["coop_reward"], defect_reward=level["defect_reward"],
                    both_defect_penalty=level["both_defect_penalty"], repro_threshold=level["repro_threshold"],
                    repro_cost=level["repro_cost"], coop_loss=level["coop_loss"], base_production=level["base_production"],
                    newborn_coins=level["newborn_coins"])
    else:
        game = Game(mode=mode, level_name=level["name"], island_capacity=level["island_capacity"], disaster_prob=level["disaster_prob"],
                    boom_prob=level["boom_prob"], coop_reward=level["coop_reward"], defect_reward=level["defect_reward"],
                    both_defect_penalty=level["both_defect_penalty"], repro_threshold=level["repro_threshold"],
                    repro_cost=level["repro_cost"], coop_loss=level["coop_loss"], base_production=level["base_production"],
                    newborn_coins=level["newborn_coins"])

    game.run_simulation(num_rounds=500)

    print("\n最终代理状态：")
    for i, agent in enumerate(game.agents):
        print(f"代理 {i} ({agent.strategy.name}, {'玩家' if agent.is_player else '机器'})："
              f"{agent.coins} 硬币，恶名 {agent.infamy}，复仇者 {agent.is_revenger}")

if __name__ == "__main__":
    main()