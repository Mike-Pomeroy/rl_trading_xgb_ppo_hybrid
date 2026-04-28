from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (
    TRAIN_BASKET_DATA_PATH,
    PPO_MODEL_PATH,
    PPO_VECNORM_PATH,
    INITIAL_AMOUNT,
    TRANSACTION_COST,
)
from trading_env_xgb_top4 import XGBTop4TradingEnv


TOTAL_TIMESTEPS = 200000


def make_env():
    return XGBTop4TradingEnv(
        basket_csv_path=TRAIN_BASKET_DATA_PATH,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost=TRANSACTION_COST,
        include_cash=True,
    )


def main():
    print("Creating training env...")
    env = DummyVecEnv([make_env])

    print("Wrapping with VecNormalize...")
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    print("Initializing PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        device="cpu",
    )

    print("Starting training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    print(f"Saving PPO model to {PPO_MODEL_PATH}")
    model.save(PPO_MODEL_PATH)

    print(f"Saving VecNormalize to {PPO_VECNORM_PATH}")
    env.save(PPO_VECNORM_PATH)

    print("Hybrid PPO training complete.")


if __name__ == "__main__":
    main()